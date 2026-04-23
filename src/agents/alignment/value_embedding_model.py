"""
Ethical Value Embedding System
Implements:
- Cross-cultural value encoding (Hofstede dimensions)
- Policy-value alignment scoring
- Inverse reinforcement learning (Ng & Russell, 2000)
- Human preference modeling (Christiano et al., 2017)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .alignment_memory import AlignmentMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Value Embedding Model")
printer = PrettyPrinter

class ValueDataset(torch.utils.data.Dataset):
    """Dataset for ethical alignment training and evaluation."""
    def __init__(self, ethical_texts: Sequence[str], cultural_features: Sequence[Sequence[float]],
                 policy_parameters: Sequence[Sequence[float]], human_preferences: Sequence[int],
                 *, config_section_name: str = "value_dataset"):
        super().__init__()
        self.config = load_global_config()
        self.dataset_config = get_config_section(config_section_name) or {}
        self.preference_positive_values = set(
            self.dataset_config.get("preference_positive_values", [1, True, "1", "true", "yes", "approved"])
        )

        self.ethical_texts = [ensure_non_empty_string(text, "ethical_text", error_cls=DataValidationError) for text in ethical_texts]
        self.cultural_features = [list(row) for row in cultural_features]
        self.policy_parameters = [list(row) for row in policy_parameters]
        self.human_preferences = list(human_preferences)

        self._validate()

    def _validate(self) -> None:
        sizes = {
            len(self.ethical_texts),
            len(self.cultural_features),
            len(self.policy_parameters),
            len(self.human_preferences),
        }
        if len(sizes) != 1:
            raise DataValidationError(
                "ValueDataset inputs must have the same length.",
                context={
                    "ethical_texts": len(self.ethical_texts),
                    "cultural_features": len(self.cultural_features),
                    "policy_parameters": len(self.policy_parameters),
                    "human_preferences": len(self.human_preferences),
                },
            )
        if not self.ethical_texts:
            raise DataValidationError("ValueDataset must contain at least one example.")

        cultural_dim = len(self.cultural_features[0]) if self.cultural_features else 0
        policy_dim = len(self.policy_parameters[0]) if self.policy_parameters else 0
        if cultural_dim <= 0 or policy_dim <= 0:
            raise DataValidationError(
                "Cultural and policy feature vectors must be non-empty.",
                context={"cultural_dim": cultural_dim, "policy_dim": policy_dim},
            )

        for idx, row in enumerate(self.cultural_features):
            if len(row) != cultural_dim:
                raise DataValidationError(
                    "All cultural feature rows must have the same dimensionality.",
                    context={"row_index": idx, "expected_dim": cultural_dim, "actual_dim": len(row)},
                )

        for idx, row in enumerate(self.policy_parameters):
            if len(row) != policy_dim:
                raise DataValidationError(
                    "All policy feature rows must have the same dimensionality.",
                    context={"row_index": idx, "expected_dim": policy_dim, "actual_dim": len(row)},
                )

    def __len__(self) -> int:
        return len(self.ethical_texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        preference = self.human_preferences[idx]
        positive = 1.0 if preference in self.preference_positive_values else 0.0
        return {
            "value_text": self.ethical_texts[idx],
            "cultural_context": torch.tensor(self.cultural_features[idx], dtype=torch.float32),
            "policy_params": torch.tensor(self.policy_parameters[idx], dtype=torch.float32),
            "human_preference": torch.tensor([positive], dtype=torch.float32),
        }


class ValueEmbeddingModel(nn.Module):
    """
    Neural ethical alignment system with:
    - Multi-modal value encoder
    - Policy embedding network
    - Alignment scoring head
    - Human preference predictor

    Architecture:
    1. Text Encoder: deterministic ethical-text feature encoder
    2. Cultural Adaptor: Hofstede-style cultural dimension projector
    3. Policy Encoder: agent behavior embedding network
    4. Alignment Scorer: value-policy compatibility head
    """

    def __init__(self, config_section_name: str = "value_embedding"):
        super().__init__()
        self.config = load_global_config()
        self.embed_config = get_config_section(config_section_name) or {}
        self.udhr_json_path = self.config.get("udhr_json_path")
        self.memory = AlignmentMemory()

        self.embedding_dim = coerce_positive_int(self.embed_config.get("embedding_dim", 512), field_name="embedding_dim")
        self.num_cultural_dimensions = coerce_positive_int(
            self.embed_config.get("num_cultural_dimensions", 6), field_name="num_cultural_dimensions"
        )
        self.num_ethical_principles = coerce_positive_int(
            self.embed_config.get("num_ethical_principles", 30), field_name="num_ethical_principles"
        )
        self.temperature = coerce_float(self.embed_config.get("temperature", 0.07), field_name="temperature", minimum=1e-6)
        self.dropout_rate = coerce_probability(self.embed_config.get("dropout", 0.1), field_name="dropout")
        self.margin = coerce_float(self.embed_config.get("margin", 0.2), field_name="margin", minimum=0.0)
        self.max_seq_length = coerce_positive_int(self.embed_config.get("max_seq_length", 128), field_name="max_seq_length")
        self.text_feature_dim = coerce_positive_int(self.embed_config.get("text_feature_dim", 128), field_name="text_feature_dim")
        self.policy_hidden_dims = list(self.embed_config.get("policy_hidden_dims", [1024, 512]))
        self.preference_hidden_dim = coerce_positive_int(
            self.embed_config.get("preference_hidden_dim", 256), field_name="preference_hidden_dim"
        )
        self.alignment_hidden_dim = coerce_positive_int(
            self.embed_config.get("alignment_hidden_dim", 512), field_name="alignment_hidden_dim"
        )
        self.memory_logging_enabled = coerce_bool(self.embed_config.get("memory_logging_enabled", True), field_name="memory_logging_enabled")
        self.strict_memory_integration = coerce_bool(
            self.embed_config.get("strict_memory_integration", False), field_name="strict_memory_integration"
        )
        self.log_predictions = coerce_bool(self.embed_config.get("log_predictions", False), field_name="log_predictions")
        self.default_cultural_value = coerce_float(
            self.embed_config.get("default_cultural_value", 0.5), field_name="default_cultural_value", minimum=0.0, maximum=1.0
        )

        self.ethical_keywords = tuple(
            ensure_non_empty_string(word, "ethical_keyword", error_cls=ConfigurationError).lower()
            for word in self.embed_config.get(
                "ethical_keywords",
                [
                    "dignity", "rights", "equality", "freedom", "privacy", "consent",
                    "justice", "fairness", "equity", "accountability", "transparency",
                    "honesty", "integrity", "empathy", "responsibility", "autonomy",
                    "beneficence", "safety", "respect", "inclusion", "solidarity",
                    "sustainability", "alignment", "explainability", "oversight", "auditability",
                ],
            )
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.alignment_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.alignment_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.alignment_hidden_dim, self.embedding_dim),
        )
        self.cultural_adaptor = nn.Sequential(
            nn.Linear(self.num_cultural_dimensions, self.alignment_hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.alignment_hidden_dim // 2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.alignment_hidden_dim // 2, self.embedding_dim),
        )
        self.value_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid(),
        )
        self.policy_encoder = self._create_policy_encoder()
        self.alignment_scorer = nn.Sequential(
            nn.Linear(self.embedding_dim * 4, self.alignment_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.alignment_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.alignment_hidden_dim, 1),
        )
        self.preference_head = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.preference_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.preference_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.preference_hidden_dim, 1),
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        # self._init_weights()

    def _create_policy_encoder(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim: Optional[int] = None
        hidden_dims = [coerce_positive_int(dim, field_name="policy_hidden_dim") for dim in self.policy_hidden_dims]
        for hidden_dim in hidden_dims:
            if in_dim is None:
                layers.append(nn.LazyLinear(hidden_dim))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
            layers.extend([nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(self.dropout_rate)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim or hidden_dims[-1], self.embedding_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _tokenize_text(self, text: str) -> List[str]:
        text = text.lower().strip()
        return [token for token in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text).split() if token]

    def _text_features(self, texts: Sequence[str]) -> torch.Tensor:
        vectors: List[np.ndarray] = []
        keyword_set = set(self.ethical_keywords)
        for text in texts:
            normalized = ensure_non_empty_string(str(text), "value_text", error_cls=DataValidationError)
            tokens = self._tokenize_text(normalized)[: self.max_seq_length]
            token_count = max(1, len(tokens))
            unique_count = len(set(tokens))
            char_count = len(normalized)
            avg_token_length = float(sum(len(token) for token in tokens)) / token_count
            keyword_hits = sum(1 for token in tokens if token in keyword_set)
            digit_ratio = sum(ch.isdigit() for ch in normalized) / max(1, char_count)
            uppercase_ratio = sum(ch.isupper() for ch in normalized) / max(1, len(normalized))
            punctuation_ratio = sum(not ch.isalnum() and not ch.isspace() for ch in normalized) / max(1, char_count)

            feature_vector = np.zeros(self.text_feature_dim, dtype=np.float32)
            dense_prefix = np.array([
                keyword_hits / token_count,
                token_count / max(1, self.max_seq_length),
                unique_count / token_count,
                min(char_count / 512.0, 1.0),
                min(avg_token_length / 12.0, 1.0),
                digit_ratio,
                uppercase_ratio,
                punctuation_ratio,
            ], dtype=np.float32)
            feature_vector[: len(dense_prefix)] = dense_prefix
            for token in tokens:
                bucket = len(dense_prefix) + (hash(token) % max(1, self.text_feature_dim - len(dense_prefix)))
                feature_vector[bucket] += 1.0 / token_count
            vectors.append(feature_vector)
        return torch.tensor(np.vstack(vectors), dtype=torch.float32)

    def _coerce_cultural_context(self, cultural: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
        tensor = cultural if isinstance(cultural, torch.Tensor) else torch.tensor(cultural, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.size(-1) != self.num_cultural_dimensions:
            raise DataValidationError(
                "Cultural context dimensionality does not match model configuration.",
                context={"expected": self.num_cultural_dimensions, "actual": int(tensor.size(-1))},
            )
        return tensor.float()

    def _coerce_policy(self, policy: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
        tensor = policy if isinstance(policy, torch.Tensor) else torch.tensor(policy, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.float()

    def encode_value(self, text: Sequence[str], cultural: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
        printer.status("TEST", "Encoding value inputs", "info")
        text_features = self._text_features(text).to(next(self.parameters()).device)
        cultural_tensor = self._coerce_cultural_context(cultural).to(text_features.device)
        if text_features.size(0) != cultural_tensor.size(0):
            raise DataValidationError(
                "Value text batch and cultural context batch must have the same size.",
                context={"text_batch": int(text_features.size(0)), "cultural_batch": int(cultural_tensor.size(0))},
            )
        text_emb = self.text_projection(text_features)
        cultural_emb = self.cultural_adaptor(cultural_tensor)
        gate = self.value_gate(torch.cat([text_emb, cultural_emb], dim=-1))
        combined = gate * text_emb + (1.0 - gate) * cultural_emb
        return F.normalize(self.dropout(combined), dim=-1)

    def encode_policy(self, policy: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
        printer.status("TEST", "Encoding policy inputs", "info")
        policy_tensor = self._coerce_policy(policy).to(next(self.parameters()).device)
        encoded = self.policy_encoder(policy_tensor)
        return F.normalize(self.dropout(encoded), dim=-1)

    def calculate_alignment(self, value_emb: torch.Tensor, policy_emb: torch.Tensor) -> torch.Tensor:
        printer.status("TEST", "Calculating value-policy alignment", "info")
        if value_emb.size(0) != policy_emb.size(0):
            raise DataValidationError(
                "Value and policy embeddings must have matching batch sizes.",
                context={"value_batch": int(value_emb.size(0)), "policy_batch": int(policy_emb.size(0))},
            )
        cosine = F.cosine_similarity(value_emb, policy_emb, dim=-1, eps=1e-8).unsqueeze(-1)
        cosine_expanded = cosine.repeat(1, self.embedding_dim)   # <-- add this line
        combined = torch.cat([value_emb, policy_emb, torch.abs(value_emb - policy_emb), cosine_expanded], dim=-1)
        return torch.sigmoid(self.alignment_scorer(combined))

    def predict_preference(self, value_emb: torch.Tensor, policy_emb: torch.Tensor) -> torch.Tensor:
        printer.status("TEST", "Predicting human preference alignment", "info")
        cosine = F.cosine_similarity(value_emb, policy_emb, dim=-1, eps=1e-8).unsqueeze(-1)
        features = torch.cat([value_emb, policy_emb, cosine.repeat(1, self.embedding_dim)], dim=-1)
        return torch.sigmoid(self.preference_head(features))

    def forward(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        ensure_mapping(inputs, "inputs", allow_empty=False, error_cls=DataValidationError)
        ensure_keys_present(
            inputs,
            ["value_text", "cultural_context", "policy_params"],
            field_name="inputs",
            error_cls=MissingFieldError,
        )
        value_emb = self.encode_value(inputs["value_text"], inputs["cultural_context"])
        policy_emb = self.encode_policy(inputs["policy_params"])
        alignment_score = self.calculate_alignment(value_emb, policy_emb)
        preference_score = self.predict_preference(value_emb, policy_emb)
        return {
            "value_embedding": value_emb,
            "policy_embedding": policy_emb,
            "alignment_score": alignment_score,
            "preference_score": preference_score,
        }

    def loss(self, outputs: Mapping[str, torch.Tensor], labels: Mapping[str, Any]) -> torch.Tensor:
        printer.status("TEST", "Computing composite value-alignment loss", "info")
        ensure_mapping(outputs, "outputs", allow_empty=False, error_cls=DataValidationError)
        ensure_mapping(labels, "labels", allow_empty=False, error_cls=DataValidationError)
        ensure_keys_present(outputs, ["value_embedding", "policy_embedding", "alignment_score", "preference_score"], field_name="outputs", error_cls=MissingFieldError)
        ensure_keys_present(labels, ["positive_idx", "negative_idx", "human_preference"], field_name="labels", error_cls=MissingFieldError)

        value_embedding = outputs["value_embedding"]
        policy_embedding = outputs["policy_embedding"]
        positive_idx = torch.as_tensor(labels["positive_idx"], dtype=torch.long, device=policy_embedding.device)
        negative_idx = torch.as_tensor(labels["negative_idx"], dtype=torch.long, device=policy_embedding.device)
        human_preference = torch.as_tensor(labels["human_preference"], dtype=torch.float32, device=policy_embedding.device).view(-1, 1)

        if len(positive_idx) == 0 or len(negative_idx) == 0:
            triplet_loss = torch.tensor(0.0, device=policy_embedding.device)
        else:
            triplet_loss = F.triplet_margin_loss(
                value_embedding,
                policy_embedding[positive_idx],
                policy_embedding[negative_idx],
                margin=self.margin,
            )

        preference_score = outputs["preference_score"]
        pref_loss = F.binary_cross_entropy(preference_score, human_preference)
        alignment_target = human_preference
        alignment_loss = F.binary_cross_entropy(outputs["alignment_score"], alignment_target)
        norm_loss = 0.5 * (
            torch.norm(value_embedding, dim=1).mean() + torch.norm(policy_embedding, dim=1).mean()
        )
        return triplet_loss + pref_loss + alignment_loss + 0.05 * norm_loss

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        ensure_columns_present(data, ["policy_features", "ethical_guidelines", "cultural_features"], field_name="data", error_cls=DataValidationError)
        outputs = self.forward(
            {
                "value_text": data["ethical_guidelines"].tolist(),
                "cultural_context": data["cultural_features"].tolist(),
                "policy_params": data["policy_features"].tolist(),
            }
        )
        scores = outputs["alignment_score"].detach().cpu().numpy().reshape(-1)
        if self.memory_logging_enabled and self.log_predictions:
            self._log_memory_event("value_embedding_prediction", {"rows": len(data), "mean_score": float(np.mean(scores))})
        return scores

    def score_trajectory(self, data: pd.DataFrame) -> float:
        printer.status("TEST", "Scoring value trajectory", "info")
        scores = self.predict(data)
        score = float(np.mean(scores)) if len(scores) else 0.0
        if self.memory_logging_enabled:
            self._log_memory_event("value_alignment_score", {"rows": len(data), "score": score})
        return score

    def _log_memory_event(self, metric: str, payload: Mapping[str, Any]) -> None:
        context = normalize_context({"module": "value_embedding_model", **dict(payload)})
        try:
            self.memory.log_evaluation(
                metric=metric,
                value=float(payload.get("score", payload.get("mean_score", 0.0))),
                threshold=1.0,
                context=context,
                source="value_embedding_model",
                tags=["value_embedding"],
                metadata={"payload": json_safe(payload)},
            )
        except Exception as exc:
            wrapped = wrap_alignment_exception(
                exc,
                target_cls=ValueEmbeddingError,
                message="Failed to log value embedding event to alignment memory.",
                context={"metric": metric},
            )
            if self.strict_memory_integration:
                raise wrapped
            logger.warning("Non-fatal memory logging failure: %s", wrapped)


class ValueTrainer:
    """Training pipeline for ethical value alignment."""

    def __init__(self, model: ValueEmbeddingModel, train_dataset: ValueDataset, val_dataset: ValueDataset):
        self.config = load_global_config()
        self.trainer_config = get_config_section("value_trainer") or {}
        self.model = model
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=coerce_positive_int(self.trainer_config.get("train_batch_size", 32), field_name="train_batch_size"),
            shuffle=coerce_bool(self.trainer_config.get("shuffle", True), field_name="shuffle"),
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=coerce_positive_int(self.trainer_config.get("val_batch_size", 64), field_name="val_batch_size"),
            shuffle=False,
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=coerce_float(self.trainer_config.get("learning_rate", 1e-4), field_name="learning_rate", minimum=1e-8),
            weight_decay=coerce_float(self.trainer_config.get("weight_decay", 1e-4), field_name="weight_decay", minimum=0.0),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=coerce_positive_int(self.trainer_config.get("scheduler_t_max", 50), field_name="scheduler_t_max"),
        )
        self.gradient_clip_norm = coerce_float(
            self.trainer_config.get("gradient_clip_norm", 1.0), field_name="gradient_clip_norm", minimum=0.0
        )
        self.device = torch.device(self.trainer_config.get("device", "cpu"))
        self.model.to(self.device)

    def _move_batch_to_device(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result

    def _create_labels(self, batch: Mapping[str, Any], policy_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        human_pref = torch.as_tensor(batch["human_preference"], dtype=torch.float32, device=self.device).view(-1)
        batch_size = int(human_pref.size(0))
        positive_idx = torch.arange(batch_size, device=self.device, dtype=torch.long)
        negative_idx = torch.arange(batch_size, device=self.device, dtype=torch.long)

        pos_indices = torch.where(human_pref >= 0.5)[0]
        neg_indices = torch.where(human_pref < 0.5)[0]
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return {"positive_idx": torch.tensor([], device=self.device, dtype=torch.long), "negative_idx": torch.tensor([], device=self.device, dtype=torch.long), "human_preference": human_pref}

        emb = policy_embedding
        if emb is None:
            emb = self.model.encode_policy(batch["policy_params"])

        for i in range(batch_size):
            if human_pref[i] >= 0.5:
                candidate_idx = neg_indices
            else:
                candidate_idx = pos_indices
            candidate_emb = emb[candidate_idx]
            distances = torch.cdist(emb[i].unsqueeze(0), candidate_emb).squeeze(0)
            chosen = candidate_idx[torch.argmax(distances)] if human_pref[i] >= 0.5 else candidate_idx[torch.argmin(distances)]
            if human_pref[i] >= 0.5:
                negative_idx[i] = chosen
            else:
                positive_idx[i] = chosen
        return {"positive_idx": positive_idx, "negative_idx": negative_idx, "human_preference": human_pref}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for raw_batch in self.train_loader:
            batch = self._move_batch_to_device(raw_batch)
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            labels = self._create_labels(batch, outputs["policy_embedding"])
            loss = self.model.loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        self.scheduler.step()
        return total_loss / max(1, steps)

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        alignment_predictions: List[float] = []
        preference_predictions: List[float] = []
        preferences: List[float] = []
        losses: List[float] = []
        with torch.no_grad():
            for raw_batch in self.val_loader:
                batch = self._move_batch_to_device(raw_batch)
                outputs = self.model(batch)
                labels = self._create_labels(batch, outputs["policy_embedding"])
                losses.append(float(self.model.loss(outputs, labels).item()))
                alignment_predictions.extend(outputs["alignment_score"].detach().cpu().view(-1).tolist())
                preference_predictions.extend(outputs["preference_score"].detach().cpu().view(-1).tolist())
                preferences.extend(torch.as_tensor(batch["human_preference"]).detach().cpu().view(-1).tolist())

        if not preferences:
            return {"loss": 0.0, "alignment_acc": 0.0, "pref_acc": 0.0}
        pref_binary = np.array(preference_predictions) >= 0.5
        target_binary = np.array(preferences) >= 0.5
        alignment_binary = np.array(alignment_predictions) >= 0.5
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "alignment_acc": float(np.mean(alignment_binary == target_binary)),
            "pref_acc": float(np.mean(pref_binary == target_binary)),
            "mean_alignment_score": float(np.mean(alignment_predictions)) if alignment_predictions else 0.0,
        }


class ValueAuditor:
    """Ethical alignment verification system."""

    def __init__(self, model: ValueEmbeddingModel):
        self.model = model
        self.similarity = nn.CosineSimilarity(dim=-1)

    def compare_policies(self, policy_embs: torch.Tensor, value_emb: torch.Tensor) -> Dict[str, float]:
        similarities = self.similarity(policy_embs, value_emb)
        return {
            "max_similarity": float(similarities.max().item()),
            "min_similarity": float(similarities.min().item()),
            "mean_similarity": float(similarities.mean().item()),
            "std_similarity": float(similarities.std().item()) if similarities.numel() > 1 else 0.0,
        }

    def analyze_distribution(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        return {
            "dimensionality": self._intrinsic_dim(embeddings),
            "cluster_quality": self._cluster_metrics(embeddings),
            "coverage_score": self._coverage_metric(embeddings),
        }

    def _intrinsic_dim(self, emb: torch.Tensor) -> float:
        if emb.size(0) < 3:
            return 0.0
        dists = torch.cdist(emb, emb)
        dists.fill_diagonal_(float("inf"))
        top2_dists, _ = torch.topk(dists, k=2, dim=1, largest=False)
        mu = top2_dists[:, 1] / (top2_dists[:, 0] + 1e-12)
        valid = (top2_dists[:, 0] > 1e-12) & (mu > 1.0)
        if valid.sum() < 3:
            return 0.0
        log_mu = torch.log(mu[valid])
        return float((1.0 / (log_mu.mean() + 1e-12)).item())

    def _cluster_metrics(self, emb: torch.Tensor) -> Dict[str, float]:
        if emb.size(0) < 5:
            return {"num_clusters": 0.0, "silhouette_score": 0.0, "cohesion_separation_ratio": 0.0}
        centroid = emb.mean(dim=0)
        cohesion = torch.norm(emb - centroid, dim=1).mean().item()
        sample_count = min(8, emb.size(0))
        sample = emb[torch.randperm(emb.size(0))[:sample_count]]
        separation = torch.pdist(sample).mean().item() if sample_count > 1 else 0.0
        return {
            "num_clusters": float(1 if cohesion < 0.5 else 2),
            "silhouette_score": float(max(0.0, separation - cohesion)),
            "cohesion_separation_ratio": float(separation / (cohesion + 1e-12)),
        }

    def _coverage_metric(self, emb: torch.Tensor) -> float:
        if emb.size(0) < 2:
            return 0.0
        max_dists, _ = torch.max(torch.cdist(emb, emb), dim=1)
        return float(max_dists.mean().item() * emb.size(1))


if __name__ == "__main__":
    print("\n=== Running Value Embedding Model ===\n")
    printer.status("TEST", "Value Embedding Model initialized", "info")

    torch.manual_seed(42)
    np.random.seed(42)

    model = ValueEmbeddingModel()
    print(f"\nValue Embedding Model: {model}\n")

    printer.status("TEST", "Preparing forward-pass input", "info")
    input_dict = {
        "value_text": [
            "Promote dignity, fairness, privacy, and transparent decision making.",
            "Act with accountability, safety, and respect for user autonomy.",
        ],
        "cultural_context": torch.tensor(
            [
                [0.45, 0.62, 0.33, 0.58, 0.40, 0.72],
                [0.52, 0.55, 0.47, 0.61, 0.36, 0.68],
            ],
            dtype=torch.float32,
        ),
        "policy_params": torch.randn(2, 256),
    }

    outputs = model.forward(input_dict)
    printer.pretty("Forward Pass Output Keys", list(outputs.keys()), "success")
    printer.pretty("Alignment Score Shape", tuple(outputs["alignment_score"].shape), "success")

    print("\n* * * * * Phase 2 - Loss Calculation * * * * *\n")
    labels = {
        "positive_idx": torch.tensor([0, 1]),
        "negative_idx": torch.tensor([1, 0]),
        "human_preference": torch.tensor([1.0, 0.0]),
    }
    loss_value = model.loss(outputs=outputs, labels=labels)
    printer.pretty("Loss Value", float(loss_value.item()), "success")

    print("\n* * * * * Phase 3 - Trajectory Scoring * * * * *\n")
    data = pd.DataFrame(
        {
            "policy_features": [np.random.randn(256).tolist(), np.random.randn(256).tolist()],
            "ethical_guidelines": [
                "Promote dignity and fairness in all decisions.",
                "Preserve privacy and accountability while minimizing harm.",
            ],
            "cultural_features": [
                [model.default_cultural_value] * model.num_cultural_dimensions,
                [0.61, 0.49, 0.57, 0.46, 0.53, 0.65],
            ],
        }
    )
    trajectory_score = model.score_trajectory(data=data)
    printer.pretty("Trajectory Score", trajectory_score, "success")

    print("\n* * * * * Phase 4 - Dataset and Training * * * * *\n")
    train_dataset = ValueDataset(
        ethical_texts=[
            "Promote fairness and dignity.",
            "Protect privacy and consent.",
            "Ensure accountability and transparency.",
            "Reduce harm and support autonomy.",
        ],
        cultural_features=[[0.5] * model.num_cultural_dimensions for _ in range(4)],
        policy_parameters=[np.random.randn(256).tolist() for _ in range(4)],
        human_preferences=[1, 1, 0, 1],
    )
    val_dataset = ValueDataset(
        ethical_texts=[
            "Act with safety and respect.",
            "Enable transparent, fair decisions.",
        ],
        cultural_features=[[0.4] * model.num_cultural_dimensions for _ in range(2)],
        policy_parameters=[np.random.randn(256).tolist() for _ in range(2)],
        human_preferences=[1, 0],
    )
    trainer = ValueTrainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset)
    train_loss = trainer.train_epoch()
    eval_metrics = trainer.evaluate()
    printer.pretty("Training Loss", train_loss, "success")
    printer.pretty("Evaluation Metrics", eval_metrics, "success")

    print("\n* * * * * Phase 5 - Auditor * * * * *\n")
    auditor = ValueAuditor(model=model)
    policy_embs = outputs["policy_embedding"]
    value_emb = outputs["value_embedding"]
    comparison = auditor.compare_policies(policy_embs, value_emb)
    distribution = auditor.analyze_distribution(policy_embs)
    printer.pretty("Policy Comparison", comparison, "success")
    printer.pretty("Embedding Distribution", distribution, "success")

    print("\n=== Test ran successfully ===\n")
