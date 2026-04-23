from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .config_loader import load_global_config, get_config_section
from .evaluation_errors import (ConfigLoadError, MetricCalculationError,
                                OperationalError, ValidationFailureError)
from ...base.modules.base_transformer import BaseTransformer
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Evaluation Transformer")
printer = PrettyPrinter


@dataclass(slots=True)
class EvaluationTransformerOutput:
    """Structured output returned by :class:`EvaluationTransformer`."""

    encoded_sequence: torch.Tensor
    pooled_representation: torch.Tensor
    reconstructed_sequence: torch.Tensor
    score_logits: torch.Tensor
    class_logits: Optional[torch.Tensor] = None
    risk_logits: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "encoded_sequence": self.encoded_sequence,
            "pooled_representation": self.pooled_representation,
            "reconstructed_sequence": self.reconstructed_sequence,
            "score_logits": self.score_logits,
            "class_logits": self.class_logits,
            "risk_logits": self.risk_logits,
            "attention_mask": self.attention_mask,
            "padding_mask": self.padding_mask,
        }


class SinusoidalEvaluationPositionalEncoding(nn.Module):
    """Deterministic positional encoding for continuous evaluation features."""

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer.")
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError("max_len must be a positive integer.")
        if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be a float in the range [0.0, 1.0).")

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=float(dropout))

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Positional encoding expects a 3D tensor of shape (batch, seq, d_model), got {tuple(x.shape)}."
            )
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum positional length {self.max_len}."
            )
        x = x + self.pe[:, :seq_len, :].to(device=x.device, dtype=x.dtype)
        return self.dropout(x)


class EvaluationTransformer(BaseTransformer):
    """
    Specialized transformer for evaluation and assessor pipelines.

    Unlike the base sequence-to-sequence transformer, this model operates on
    continuous feature sequences such as telemetry windows, evaluator metrics,
    behavior traces, or multi-signal diagnostic embeddings.

    The class still inherits from :class:`BaseTransformer` to retain project-wide
    configuration, shared transformer utilities, serialization conventions, and
    a consistent inheritance model. The evaluation-specific stack then builds a
    dedicated encoder pathway on top of the base infrastructure.
    """

    def __init__(self, input_dim: int, seq_len: int,
        d_model: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        score_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.config = load_global_config()
        self.trans_config = get_config_section("eval_transformer") or {}
        self.config_path = self.config.get("__config_path__", "<unknown>")

        self._validate_init_arguments(input_dim=input_dim, seq_len=seq_len)

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.eval_d_model = int(d_model or self.trans_config.get("d_model", self.d_model))
        self.output_dim = int(output_dim or self.trans_config.get("output_dim", self.input_dim))
        self.score_dim = int(score_dim or self.trans_config.get("score_dim", 1))
        self.num_classes = self._resolve_num_classes(num_classes)
        self.pooling = str(self.trans_config.get("pooling", "mean")).strip().lower()
        self.use_cls_token = bool(self.trans_config.get("use_cls_token", self.pooling == "cls"))
        self.enable_risk_head = bool(self.trans_config.get("enable_risk_head", True))
        self.pre_norm = bool(self.trans_config.get("pre_norm", True))
        self.max_seq_len = int(self.trans_config.get("max_seq_len", self.seq_len))
        self.encoder_dropout = float(self.trans_config.get("dropout", self.dropout))
        self.encoder_nhead = int(self.trans_config.get("nhead", self.nhead))
        self.encoder_dim_feedforward = int(
            self.trans_config.get("dim_feedforward", self.dim_feedforward)
        )
        self.encoder_num_layers = int(
            self.trans_config.get("num_encoder_layers", self.num_encoder_layers)
        )
        self.activation_name = str(self.trans_config.get("activation", self.activation)).strip().lower()
        self.head_hidden_dim = int(
            self.trans_config.get("head_hidden_dim", max(self.eval_d_model, 128))
        )

        self._validate_configuration()

        self.input_projection = nn.Linear(self.input_dim, self.eval_d_model)
        self.input_normalization = nn.LayerNorm(self.eval_d_model)
        self.input_dropout = nn.Dropout(self.encoder_dropout)

        self.pos_encoder = SinusoidalEvaluationPositionalEncoding(
            d_model=self.eval_d_model,
            max_len=self.max_seq_len + (1 if self.use_cls_token else 0),
            dropout=self.encoder_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.eval_d_model,
            nhead=self.encoder_nhead,
            dim_feedforward=self.encoder_dim_feedforward,
            dropout=self.encoder_dropout,
            activation=self.activation_name,
            batch_first=True,
            norm_first=self.pre_norm,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.encoder_num_layers,
            norm=nn.LayerNorm(self.eval_d_model),
        )

        self.sequence_norm = nn.LayerNorm(self.eval_d_model)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.eval_d_model),
            nn.Linear(self.eval_d_model, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.encoder_dropout),
            nn.Linear(self.head_hidden_dim, self.output_dim),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.eval_d_model),
            nn.Linear(self.eval_d_model, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.encoder_dropout),
            nn.Linear(self.head_hidden_dim, self.score_dim),
        )
        self.risk_head = (
            nn.Sequential(
                nn.LayerNorm(self.eval_d_model),
                nn.Linear(self.eval_d_model, self.head_hidden_dim // 2 if self.head_hidden_dim > 1 else 1),
                nn.GELU(),
                nn.Dropout(self.encoder_dropout),
                nn.Linear(self.head_hidden_dim // 2 if self.head_hidden_dim > 1 else 1, 1),
            )
            if self.enable_risk_head
            else None
        )
        self.classification_head = (
            nn.Sequential(
                nn.LayerNorm(self.eval_d_model),
                nn.Linear(self.eval_d_model, self.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.encoder_dropout),
                nn.Linear(self.head_hidden_dim, self.num_classes),
            )
            if self.num_classes is not None
            else None
        )

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, self.eval_d_model)) if self.use_cls_token else None
        )

        self._init_weights()
        logger.info(
            "EvaluationTransformer initialized: input_dim=%d seq_len=%d eval_d_model=%d layers=%d nhead=%d",
            self.input_dim,
            self.seq_len,
            self.eval_d_model,
            self.encoder_num_layers,
            self.encoder_nhead,
        )

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------

    def _validate_init_arguments(self, input_dim: int, seq_len: int) -> None:
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValidationFailureError("input_dim", input_dim, "positive integer")
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValidationFailureError("seq_len", seq_len, "positive integer")

    def _resolve_num_classes(self, num_classes: Optional[int]) -> Optional[int]:
        configured = num_classes if num_classes is not None else self.trans_config.get("num_classes")
        if configured is None:
            return None
        if not isinstance(configured, int) or configured <= 1:
            raise ConfigLoadError(
                str(self.config_path),
                "eval_transformer.num_classes",
                "num_classes must be an integer greater than 1 when classification is enabled.",
            )
        return configured

    def _validate_configuration(self) -> None:
        numeric_rules = {
            "eval_d_model": self.eval_d_model,
            "output_dim": self.output_dim,
            "score_dim": self.score_dim,
            "max_seq_len": self.max_seq_len,
            "encoder_nhead": self.encoder_nhead,
            "encoder_dim_feedforward": self.encoder_dim_feedforward,
            "encoder_num_layers": self.encoder_num_layers,
            "head_hidden_dim": self.head_hidden_dim,
        }
        for name, value in numeric_rules.items():
            if not isinstance(value, int) or value <= 0:
                raise ConfigLoadError(
                    str(self.config_path),
                    f"eval_transformer.{name}",
                    f"{name} must be a positive integer, received {value!r}.",
                )

        if self.eval_d_model % self.encoder_nhead != 0:
            raise ConfigLoadError(
                str(self.config_path),
                "eval_transformer.nhead",
                "d_model must be divisible by nhead for multi-head attention.",
            )
        if self.max_seq_len < self.seq_len:
            raise ConfigLoadError(
                str(self.config_path),
                "eval_transformer.max_seq_len",
                "max_seq_len must be greater than or equal to the instantiated sequence length.",
            )
        if not 0.0 <= self.encoder_dropout < 1.0:
            raise ConfigLoadError(
                str(self.config_path),
                "eval_transformer.dropout",
                "dropout must be in the range [0.0, 1.0).",
            )
        if self.pooling not in {"mean", "max", "cls", "first"}:
            raise ConfigLoadError(
                str(self.config_path),
                "eval_transformer.pooling",
                "pooling must be one of: mean, max, cls, first.",
            )
        if self.activation_name not in {"relu", "gelu"}:
            raise ConfigLoadError(
                str(self.config_path),
                "eval_transformer.activation",
                "activation must be either 'relu' or 'gelu'.",
            )

    # ------------------------------------------------------------------
    # Core encoding pipeline
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_sequence: bool = True,
    ) -> EvaluationTransformerOutput | Dict[str, Optional[torch.Tensor]]:
        """
        Encode a continuous evaluation sequence and return task heads.

        Args:
            x: Input tensor of shape ``(batch, sequence, input_dim)``.
            attention_mask: Optional mask of shape ``(batch, sequence)`` where
                truthy values indicate valid positions and falsy values indicate
                padding positions.
            return_dict: Return a serializable dict instead of the dataclass.
            return_sequence: Keep the encoded sequence in the output payload.
        """
        encoded_sequence, normalized_attention_mask, padding_mask = self.encode_features(
            x,
            attention_mask=attention_mask,
            return_masks=True,
        )

        sequence_output = encoded_sequence[:, 1:, :] if self.use_cls_token else encoded_sequence
        sequence_attention_mask = (
            normalized_attention_mask[:, 1:] if normalized_attention_mask is not None and self.use_cls_token else normalized_attention_mask
        )
        sequence_padding_mask = (
            padding_mask[:, 1:] if padding_mask is not None and self.use_cls_token else padding_mask
        )

        pooled_input = encoded_sequence if self.use_cls_token else sequence_output
        pooled_attention_mask = normalized_attention_mask if self.use_cls_token else sequence_attention_mask
        pooled = self.pool_sequence(
            pooled_input,
            attention_mask=pooled_attention_mask,
        )

        reconstructed = self.reconstruction_head(sequence_output)
        score_logits = self.score_head(pooled)
        class_logits = self.classification_head(pooled) if self.classification_head is not None else None
        risk_logits = self.risk_head(pooled) if self.risk_head is not None else None

        output = EvaluationTransformerOutput(
            encoded_sequence=sequence_output if return_sequence else sequence_output[:, -1:, :],
            pooled_representation=pooled,
            reconstructed_sequence=reconstructed,
            score_logits=score_logits,
            class_logits=class_logits,
            risk_logits=risk_logits,
            attention_mask=sequence_attention_mask,
            padding_mask=sequence_padding_mask,
        )
        return output.to_dict() if return_dict else output

    def encode_features(self, x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_masks: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Project and encode continuous input features using the evaluation encoder."""
        validated = self._validate_input_tensor(x)
        normalized_attention_mask, padding_mask = self._normalize_attention_mask(
            attention_mask,
            batch_size=validated.size(0),
            seq_len=validated.size(1),
            device=validated.device,
        )

        features = self.input_projection(validated)
        features = self.input_normalization(features)
        features = self.input_dropout(features)

        if self.use_cls_token:
            features, normalized_attention_mask, padding_mask = self._prepend_cls_token(
                features,
                normalized_attention_mask,
                padding_mask,
            )

        features = self.pos_encoder(features)
        encoded = self.encoder(features, src_key_padding_mask=padding_mask)
        encoded = self.sequence_norm(encoded)

        if return_masks:
            return encoded, normalized_attention_mask, padding_mask
        return encoded

    def pool_sequence(self, encoded_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        strategy: Optional[str] = None,
    ) -> torch.Tensor:
        """Pool an encoded sequence into a single representation."""
        if encoded_sequence.dim() != 3:
            raise ValidationFailureError(
                "encoded_sequence_rank",
                tuple(encoded_sequence.shape),
                "3D tensor (batch, sequence, hidden_dim)",
            )

        pooling_strategy = (strategy or self.pooling).strip().lower()
        if pooling_strategy not in {"mean", "max", "cls", "first"}:
            raise ValidationFailureError(
                "pooling_strategy",
                pooling_strategy,
                "mean, max, cls, or first",
            )

        if pooling_strategy in {"cls", "first"}:
            return encoded_sequence[:, 0, :]

        if attention_mask is None:
            if pooling_strategy == "mean":
                return encoded_sequence.mean(dim=1)
            return encoded_sequence.max(dim=1).values

        if attention_mask.dim() != 2 or attention_mask.shape[:2] != encoded_sequence.shape[:2]:
            raise ValidationFailureError(
                "attention_mask_shape",
                tuple(attention_mask.shape),
                f"({encoded_sequence.size(0)}, {encoded_sequence.size(1)})",
            )

        valid_mask = attention_mask.to(device=encoded_sequence.device, dtype=encoded_sequence.dtype).unsqueeze(-1)
        valid_count = valid_mask.sum(dim=1).clamp_min(1.0)

        if pooling_strategy == "mean":
            return (encoded_sequence * valid_mask).sum(dim=1) / valid_count

        masked = encoded_sequence.masked_fill(valid_mask.eq(0), float("-inf"))
        pooled = masked.max(dim=1).values
        pooled[torch.isinf(pooled)] = 0.0
        return pooled

    # ------------------------------------------------------------------
    # Task helpers and losses
    # ------------------------------------------------------------------

    def reconstruct(self, x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encode_features(x, attention_mask=attention_mask)
        return self.reconstruction_head(encoded)

    def score(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded, normalized_attention_mask, _ = self.encode_features(
            x,
            attention_mask=attention_mask,
            return_masks=True,
        )
        pooled = self.pool_sequence(encoded, attention_mask=normalized_attention_mask)
        return self.score_head(pooled)

    def classify(self, x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.classification_head is None:
            raise OperationalError(
                "Classification head is not enabled for this EvaluationTransformer instance.",
                context={"num_classes": self.num_classes},
            )
        encoded, normalized_attention_mask, _ = self.encode_features(
            x,
            attention_mask=attention_mask,
            return_masks=True,
        )
        pooled = self.pool_sequence(encoded, attention_mask=normalized_attention_mask)
        return self.classification_head(pooled)

    def estimate_risk(self, x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        apply_sigmoid: bool = True,
    ) -> torch.Tensor:
        if self.risk_head is None:
            raise OperationalError(
                "Risk head is not enabled for this EvaluationTransformer instance.",
                context={"enable_risk_head": self.enable_risk_head},
            )
        encoded, normalized_attention_mask, _ = self.encode_features(
            x,
            attention_mask=attention_mask,
            return_masks=True,
        )
        pooled = self.pool_sequence(encoded, attention_mask=normalized_attention_mask)
        logits = self.risk_head(pooled)
        return torch.sigmoid(logits) if apply_sigmoid else logits

    def compute_losses(self,
        batch: Mapping[str, torch.Tensor],
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute a composite loss dictionary for training and diagnostics.

        Supported batch keys
        --------------------
        inputs:           required continuous feature tensor
        attention_mask:   optional validity mask
        reconstruction_target: optional target for reconstruction; defaults to inputs
        score_target:     optional regression target
        class_target:     optional classification target
        risk_target:      optional binary risk target
        """
        if not isinstance(batch, Mapping):
            raise ValidationFailureError("batch_type", type(batch).__name__, "mapping/dictionary")
        if "inputs" not in batch:
            raise ValidationFailureError("batch.inputs", None, "tensor under the 'inputs' key")

        inputs = batch["inputs"]
        attention_mask = batch.get("attention_mask")
        outputs = self.forward(inputs, attention_mask=attention_mask, return_dict=False)

        losses: Dict[str, torch.Tensor] = {}
        reconstruction_target = batch.get("reconstruction_target", inputs)
        losses["reconstruction"] = self._masked_mse_loss(
            outputs.reconstructed_sequence,
            reconstruction_target,
            attention_mask=outputs.attention_mask,
            reduction=reduction,
        )

        if "score_target" in batch:
            losses["score"] = F.mse_loss(outputs.score_logits, batch["score_target"], reduction=reduction)

        if outputs.class_logits is not None and "class_target" in batch:
            losses["classification"] = F.cross_entropy(outputs.class_logits, batch["class_target"])

        if outputs.risk_logits is not None and "risk_target" in batch:
            risk_target = batch["risk_target"].to(dtype=outputs.risk_logits.dtype)
            losses["risk"] = F.binary_cross_entropy_with_logits(
                outputs.risk_logits,
                risk_target,
                reduction=reduction,
            )

        losses["total"] = torch.stack(list(losses.values())).sum()
        return losses

    def summarize_batch(self, x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Return lightweight numeric diagnostics for a batch."""
        try:
            outputs = self.forward(x, attention_mask=attention_mask, return_dict=False)
            summary = {
                "batch_size": float(x.size(0)),
                "sequence_length": float(x.size(1)),
                "feature_dimension": float(x.size(2)),
                "encoded_l2_mean": float(outputs.encoded_sequence.norm(dim=-1).mean().item()),
                "score_mean": float(outputs.score_logits.mean().item()),
                "score_std": float(outputs.score_logits.std(unbiased=False).item()),
                "reconstruction_abs_mean": float(outputs.reconstructed_sequence.abs().mean().item()),
            }
            if outputs.class_logits is not None:
                summary["class_logit_mean"] = float(outputs.class_logits.mean().item())
            if outputs.risk_logits is not None:
                summary["risk_probability_mean"] = float(torch.sigmoid(outputs.risk_logits).mean().item())
            return summary
        except Exception as exc:  # noqa: BLE001 - convert arbitrary math/runtime failures into domain errors.
            raise MetricCalculationError(
                "evaluation_transformer_batch_summary",
                inputs={"shape": tuple(x.shape)},
                reason=str(exc),
            ) from exc

    def save(self, path: str) -> None:
        """Persist model weights together with evaluation-specific metadata."""
        if not isinstance(path, str) or not path.strip():
            raise ValidationFailureError("save.path", path, "non-empty path string")

        payload = {
            "state_dict": self.state_dict(),
            "config": self.get_model_config(),
            "class_name": self.__class__.__name__,
        }
        torch.save(payload, path)
        logger.info("EvaluationTransformer checkpoint saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "EvaluationTransformer":
        """Load an EvaluationTransformer checkpoint with constructor metadata."""
        checkpoint = torch.load(path, map_location=device)
        if not isinstance(checkpoint, Mapping):
            raise OperationalError(
                "EvaluationTransformer checkpoint is malformed.",
                context={"path": path, "type": type(checkpoint).__name__},
            )

        config = checkpoint.get("config")
        if not isinstance(config, Mapping):
            raise OperationalError(
                "EvaluationTransformer checkpoint is missing model configuration.",
                context={"path": path},
            )

        model = cls(**dict(config))
        model.load_state_dict(checkpoint["state_dict"])
        if device is not None:
            model = model.to(device)
        logger.info("EvaluationTransformer checkpoint loaded from %s", path)
        return model

    def get_model_config(self) -> Dict[str, Any]:
        """Return the constructor metadata required to reload the model."""
        return {
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "d_model": self.eval_d_model,
            "output_dim": self.output_dim,
            "num_classes": self.num_classes,
            "score_dim": self.score_dim,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValidationFailureError("inputs.type", type(x).__name__, "torch.Tensor")
        if x.dim() != 3:
            raise ValidationFailureError("inputs.rank", x.dim(), "3 dimensions: (batch, sequence, features)")
        if x.size(-1) != self.input_dim:
            raise ValidationFailureError("inputs.feature_dim", int(x.size(-1)), self.input_dim)
        if x.size(1) > self.max_seq_len:
            raise ValidationFailureError("inputs.sequence_length", int(x.size(1)), f"<= {self.max_seq_len}")
        return x.to(dtype=torch.float32)

    def _normalize_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            return None, None
        if not isinstance(attention_mask, torch.Tensor):
            raise ValidationFailureError("attention_mask.type", type(attention_mask).__name__, "torch.Tensor")
        if attention_mask.dim() != 2 or attention_mask.shape != (batch_size, seq_len):
            raise ValidationFailureError(
                "attention_mask.shape",
                tuple(attention_mask.shape),
                (batch_size, seq_len),
            )

        if attention_mask.dtype == torch.bool:
            normalized = attention_mask.to(device=device)
        else:
            normalized = attention_mask.to(device=device)
            unique_values = set(normalized.detach().flatten().tolist())
            if not unique_values.issubset({0, 1, 0.0, 1.0, True, False}):
                raise ValidationFailureError(
                    "attention_mask.values",
                    sorted(unique_values)[:10],
                    "binary mask containing only 0/1 values",
                )
            normalized = normalized.bool()

        if normalized.sum(dim=1).eq(0).any():
            raise ValidationFailureError(
                "attention_mask.valid_tokens",
                "row contains zero valid tokens",
                "at least one valid token per sequence",
            )

        padding_mask = ~normalized
        return normalized, padding_mask

    def _prepend_cls_token(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.cls_token is None:
            return features, attention_mask, padding_mask

        batch_size = features.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1).to(dtype=features.dtype, device=features.device)
        features = torch.cat([cls_token, features], dim=1)

        if attention_mask is not None:
            cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=features.device)
            attention_mask = torch.cat([cls_valid, attention_mask], dim=1)
        if padding_mask is not None:
            cls_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=features.device)
            padding_mask = torch.cat([cls_padding, padding_mask], dim=1)

        return features, attention_mask, padding_mask

    def _masked_mse_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        reduction: str,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValidationFailureError(
                "reconstruction_target.shape",
                tuple(target.shape),
                tuple(prediction.shape),
            )

        squared_error = (prediction - target) ** 2
        if attention_mask is None:
            return squared_error.mean() if reduction == "mean" else squared_error.sum()

        mask = attention_mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
        denom = mask.sum() * prediction.size(-1)
        denom = denom.clamp_min(1.0)
        masked = squared_error * mask
        if reduction == "sum":
            return masked.sum()
        return masked.sum() / denom


if __name__ == "__main__":
    print("\n=== Running Evaluation Transformer ===\n")

    model = EvaluationTransformer(
        input_dim=32,
        seq_len=16,
        d_model=128,
        num_classes=4,
        score_dim=2,
    )

    batch = torch.randn(3, 16, 32)
    mask = torch.ones(3, 16, dtype=torch.bool)
    mask[2, -4:] = False

    outputs = model(batch, attention_mask=mask, return_dict=False)
    printer.pretty("Encoded shape", tuple(outputs.encoded_sequence.shape), "success")
    printer.pretty("Pooled shape", tuple(outputs.pooled_representation.shape), "success")
    printer.pretty("Reconstruction shape", tuple(outputs.reconstructed_sequence.shape), "success")
    printer.pretty("Score logits shape", tuple(outputs.score_logits.shape), "success")
    if outputs.class_logits is not None:
        printer.pretty("Class logits shape", tuple(outputs.class_logits.shape), "success")
    if outputs.risk_logits is not None:
        printer.pretty("Risk logits shape", tuple(outputs.risk_logits.shape), "success")

    losses = model.compute_losses(
        {
            "inputs": batch,
            "attention_mask": mask,
            "score_target": torch.randn(3, 2),
            "class_target": torch.tensor([0, 1, 2]),
            "risk_target": torch.tensor([[0.0], [1.0], [0.0]]),
        }
    )
    printer.pretty(
        "Loss keys",
        {key: float(value.detach().cpu().item()) for key, value in losses.items()},
        "success",
    )
    printer.pretty("Batch summary", model.summarize_batch(batch, attention_mask=mask), "success")

    print("\n=== Successfully Ran Evaluation Transformer ===\n")
