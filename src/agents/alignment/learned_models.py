"""Optional learned preference evidence built on SLAI's existing value embedding stack.

The encoder and scorer are trainable separately. Neither interprets laws, grants
permissions, or decides whether an action may execute. Model artifacts are
untrusted until their manifest and tensor-only checkpoint hashes are checked.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .value_embedding_model import ValueEmbeddingModel

SCHEMA = "slai.alignment.learned.v1"
FEATURE_SCHEMA = "value_text.blake2b_v1"
ENCODER_FILE = "representation_encoder.pt"
SCORER_FILE = "value_alignment_scorer.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_texts(texts: Sequence[str], *, field: str) -> Tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
        raise ValueError(f"{field} must be a non-empty sequence of strings")
    normalized = []
    for item in texts:
        if not isinstance(item, str) or not item.strip() or len(item) > 8192:
            raise ValueError(f"{field} contains an empty, oversized or invalid string")
        normalized.append(item.strip())
    return tuple(normalized)


class RepresentationEncoder(nn.Module):
    """Trainable text representation, reusing ValueEmbeddingModel's text stack.

    Only the shared text projection is trained/saved. Unused policy/ethics heads
    from the pre-existing model are not included in the encoder checkpoint.
    """

    def __init__(self, backbone: Optional[ValueEmbeddingModel] = None) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else ValueEmbeddingModel()
        if self.backbone.text_hash_strategy != "blake2b_v1":
            raise ValueError("RepresentationEncoder requires stable blake2b_v1 text features")
        self.embedding_dim = self.backbone.embedding_dim

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        validated = _validate_texts(texts, field="texts")
        features = self.backbone._text_features(validated)
        device = next(self.backbone.text_projection.parameters()).device
        vectors = self.backbone.text_projection(features.to(device))
        return F.normalize(vectors, p=2, dim=-1, eps=1e-8)

    def trainable_parameters(self):
        return self.backbone.text_projection.parameters()

    def features_manifest(self) -> Dict[str, Any]:
        b = self.backbone
        keywords = json.dumps(list(b.ethical_keywords), separators=(",", ":"), ensure_ascii=False)
        return {
            "feature_schema": FEATURE_SCHEMA,
            "embedding_dim": b.embedding_dim,
            "text_feature_dim": b.text_feature_dim,
            "max_seq_length": b.max_seq_length,
            "ethical_keywords_sha256": hashlib.sha256(keywords.encode("utf-8")).hexdigest(),
        }

    def export_weights(self) -> Dict[str, torch.Tensor]:
        return {
            key: tensor.detach().cpu().contiguous()
            for key, tensor in self.backbone.text_projection.state_dict().items()
        }

    def import_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        self.backbone.text_projection.load_state_dict(dict(weights), strict=True)


class ValueAlignmentScorer(nn.Module):
    """Learned preference compatibility, using the SAME encoder representations.

    A score is an uncalibrated compatibility signal, not policy clearance.
    """

    def __init__(self, encoder: RepresentationEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        # Reuse the existing, trainable value-policy compatibility head.
        self.head = encoder.backbone.alignment_scorer

    def forward_logits(
        self, value_texts: Sequence[str], action_texts: Sequence[str]
    ) -> torch.Tensor:
        if len(value_texts) != len(action_texts):
            raise ValueError("Value and action batch sizes must match")
        value_emb = self.encoder(value_texts)
        action_emb = self.encoder(action_texts)
        similarity = F.cosine_similarity(value_emb, action_emb, dim=-1, eps=1e-8)
        inputs = torch.cat(
            [
                value_emb,
                action_emb,
                torch.abs(value_emb - action_emb),
                similarity.unsqueeze(-1).expand(-1, self.encoder.embedding_dim),
            ],
            dim=-1,
        )
        return self.head(inputs).reshape(-1)

    def forward(
        self, value_texts: Sequence[str], action_texts: Sequence[str]
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(value_texts, action_texts))

    def export_weights(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().contiguous() for key, value in self.head.state_dict().items()}

    def import_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        self.head.load_state_dict(dict(weights), strict=True)


class AlignmentModelBundle:
    """Integrity-checked read-only inference bundle; no unsafe pickle fallback."""

    def __init__(
        self, encoder: RepresentationEncoder, scorer: ValueAlignmentScorer,
        manifest: Mapping[str, Any], *, artifact_dir: Path
    ) -> None:
        self.encoder = encoder.eval().cpu()
        self.scorer = scorer.eval().cpu()
        self.manifest = dict(manifest)
        self.artifact_dir = artifact_dir

    @classmethod
    def load(cls, artifact_dir: str | Path, *, require_approved: bool = False) -> "AlignmentModelBundle":
        root = Path(artifact_dir).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Alignment model artifact path must be a directory")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported alignment model manifest schema")
        if require_approved and manifest.get("approved") is not True:
            raise ValueError("Alignment model has not been approved for active inference")

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("Alignment manifest has no artifact hashes")
        for filename in (ENCODER_FILE, SCORER_FILE):
            expected = artifacts.get(filename)
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"Missing artifact digest: {filename}")
            actual = _sha256(root / filename)
            if actual != expected:
                raise ValueError(f"Alignment model artifact integrity failure: {filename}")

        encoder = RepresentationEncoder()
        expected_features = encoder.features_manifest()
        if manifest.get("features") != expected_features:
            raise ValueError("Alignment feature schema/configuration mismatch")
        scorer = ValueAlignmentScorer(encoder)
        # Use only tensor-only state dictionaries. Never enable unsafe pickle.
        enc_weights = torch.load(root / ENCODER_FILE, map_location="cpu", weights_only=True)
        score_weights = torch.load(root / SCORER_FILE, map_location="cpu", weights_only=True)
        if not isinstance(enc_weights, dict) or not isinstance(score_weights, dict):
            raise ValueError("Alignment checkpoint must be a tensor state_dict")
        encoder.import_weights(enc_weights)
        scorer.import_weights(score_weights)
        return cls(encoder, scorer, manifest, artifact_dir=root)

    def score(self, *, value_text: str, action_text: str) -> Dict[str, Any]:
        value = _validate_texts([value_text], field="value_text")
        action = _validate_texts([action_text], field="action_text")
        with torch.inference_mode():
            score = float(self.scorer(value, action).item())
            # Similarity is a descriptive representation metric, not agreement.
            similarity = float(
                F.cosine_similarity(self.encoder(value), self.encoder(action), dim=-1).item()
            )
        if not math.isfinite(score) or not math.isfinite(similarity):
            raise ValueError("Non-finite alignment model inference")
        return {
            "score": score,
            "representation_similarity": similarity,
            "model_version": self.manifest.get("model_version", "unknown"),
            "feature_schema": FEATURE_SCHEMA,
            "calibrated": bool(self.manifest.get("calibrated", False)),
            "approved": self.manifest.get("approved") is True,
        }


__all__ = ["RepresentationEncoder", "ValueAlignmentScorer", "AlignmentModelBundle"]
