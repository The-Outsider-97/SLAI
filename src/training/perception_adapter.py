"""Optional frozen PerceptionAgent teacher adapter for LANTRA curriculum mining.

This module never trains PerceptionAgent and never injects its tensors directly
into LANTRA. It only obtains frozen text representations to estimate semantic
hardness/alignment for already grounded curriculum candidates.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .enrichment_contracts import CurriculumCompatibilityError, CurriculumConfig, sha256_payload


class PerceptionAdapter:
    """Bounded, cached text-embedding access through PerceptionAgent.perform_task()."""

    def __init__(self, agent: Any, config: CurriculumConfig, *, cache_size: int = 4096) -> None:
        self.agent = agent
        self.config = config
        if not callable(getattr(agent, "perform_task", None)):
            raise CurriculumCompatibilityError(
                "PerceptionAgent does not expose perform_task()."
            )
        self._cache_size = max(32, int(cache_size))
        self._cache: "collections.OrderedDict[str, Tuple[float, ...]]" = collections.OrderedDict()
        self.checkpoint_metadata: Dict[str, Any] = {}
        self._prepared = False

    def prepare(self) -> Dict[str, Any]:
        if self._prepared:
            return dict(self.checkpoint_metadata)
        version = str(self.config.perception_checkpoint_version or "").strip() or None
        if version is None and self.config.require_perception_checkpoint:
            raise CurriculumCompatibilityError(
                "Perception enrichment requires a trained SLAI checkpoint version."
            )
        if version is not None:
            restore = getattr(self.agent, "restore_checkpoint", None)
            if not callable(restore):
                raise CurriculumCompatibilityError(
                    "PerceptionAgent does not expose restore_checkpoint()."
                )
            result = restore_checkpoint_safely(restore, version)
            if not isinstance(result, Mapping) or result.get("status") != "success":
                raise CurriculumCompatibilityError(
                    f"Perception checkpoint restore did not succeed: {result!r}"
                )
            self.checkpoint_metadata = dict(result)
        else:
            self.checkpoint_metadata = {
                "status": "uncheckpointed",
                "warning": "explicitly allowed by configuration",
            }
        self._prepared = True
        return dict(self.checkpoint_metadata)

    def embed_text(self, text: str) -> Tuple[float, ...]:
        self.prepare()
        key = sha256_payload({"text": text})
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        result = self.agent.perform_task(
            {
                "task_type": "inference",
                "modality": "text",
                "input_data": text,
                "return_sequence": False,
            }
        )
        if not isinstance(result, Mapping) or result.get("status") != "success":
            raise CurriculumCompatibilityError(
                f"Perception text inference failed: {result!r}"
            )
        vector = self._flatten_numeric(result.get("output"))
        if not vector:
            raise CurriculumCompatibilityError(
                "PerceptionAgent returned an empty/non-numeric text representation."
            )
        normalized = self._l2_normalize(vector)
        self._cache[key] = normalized
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return normalized

    def similarity(self, left: str, right: str) -> float:
        a = self.embed_text(left)
        b = self.embed_text(right)
        return self._cosine(a, b)

    def score_triplet(self, anchor: str, positive: str, negative: str) -> Dict[str, float]:
        anchor_vec = self.embed_text(anchor)
        positive_vec = self.embed_text(positive)
        negative_vec = self.embed_text(negative)
        positive_similarity = self._cosine(anchor_vec, positive_vec)
        negative_similarity = self._cosine(anchor_vec, negative_vec)
        return {
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "margin": positive_similarity - negative_similarity,
        }

    @classmethod
    def _flatten_numeric(cls, value: Any) -> Tuple[float, ...]:
        # Tensor-like values returned by detach_tree still expose these methods.
        current = value
        for method_name in ("detach", "cpu"):
            method = getattr(current, method_name, None)
            if callable(method):
                current = method()
        reshape = getattr(current, "reshape", None)
        if callable(reshape):
            try:
                current = reshape(-1)
            except Exception:
                pass
        tolist = getattr(current, "tolist", None)
        if callable(tolist):
            current = tolist()

        flattened: List[float] = []

        def visit(item: Any) -> None:
            if isinstance(item, bool):
                return
            if isinstance(item, (int, float)):
                number = float(item)
                if math.isfinite(number):
                    flattened.append(number)
                return
            if isinstance(item, Mapping):
                for key in sorted(item, key=str):
                    visit(item[key])
                return
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                for child in item:
                    visit(child)

        visit(current)
        return tuple(flattened)

    @staticmethod
    def _l2_normalize(vector: Sequence[float]) -> Tuple[float, ...]:
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm <= 0.0:
            raise CurriculumCompatibilityError("Perception representation has zero L2 norm.")
        return tuple(float(value) / norm for value in vector)

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            raise CurriculumCompatibilityError(
                "Perception representation dimensions are inconsistent."
            )
        # Vectors are already normalized, but clamp small numerical drift.
        value = sum(float(a) * float(b) for a, b in zip(left, right))
        return max(-1.0, min(1.0, value))


def restore_checkpoint_safely(restore_callable: Any, version: str) -> Any:
    """Call the canonical Perception checkpoint API without path heuristics."""

    return restore_callable(version, verify_integrity=True)


__all__ = ["PerceptionAdapter"]
