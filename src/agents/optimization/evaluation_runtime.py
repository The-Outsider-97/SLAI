"""Durable, version-checked evaluation evidence for deterministic optimizers.

This module is not an optimization solver or a replacement for OptimizationMemory:
it persists candidate-level evaluations and optional *algorithm progress*, whereas
OptimizationMemory stores completed run evidence and warm starts.

Checkpoint files contain JSON primitives only; never Python objects or pickles.
All callbacks must be trusted. Persistence is synchronous and single-writer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .utils.optimization_errors import OptimizationValidationError


class EvaluationJournal:
    """Crash-safe evaluation cache with explicit caller-supplied identity.

    Identity MUST capture the mathematical problem, evaluator implementation,
    authoritative model revision, and solver options that affect search progress.
    Reopening with a different identity fails closed; no stale reuse.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        identity: str,
        resume: bool = False,
        maximum_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not isinstance(identity, str) or not identity.strip():
            raise OptimizationValidationError("evaluation journal needs a nonempty identity")
        self.path = Path(path)
        if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes <= 0:
            raise OptimizationValidationError("maximum_bytes must be a positive integer")
        self.identity = identity
        self.maximum_bytes = maximum_bytes
        self.entries: dict[str, dict[str, Any]] = {}
        self.progress: dict[str, Any] | None = None
        if resume:
            self.load()

    def load(self) -> None:
        try:
            if self.path.stat().st_size > self.maximum_bytes:
                raise OptimizationValidationError("evaluation checkpoint exceeds size limit")
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise OptimizationValidationError("evaluation checkpoint is missing or invalid") from exc
        if not isinstance(data, dict) or data.get("version") != self.SCHEMA_VERSION:
            raise OptimizationValidationError("unsupported evaluation checkpoint schema")
        checksum = data.pop("sha256", None)
        try:
            encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise OptimizationValidationError("evaluation checkpoint contains non-JSON-safe data") from exc
        if not isinstance(checksum, str) or hashlib.sha256(encoded.encode("utf-8")).hexdigest() != checksum:
            raise OptimizationValidationError("evaluation checkpoint checksum mismatch")
        if data.get("identity") != self.identity:
            raise OptimizationValidationError("evaluation checkpoint identity mismatch; refusing stale reuse")
        raw = data.get("entries")
        if not isinstance(raw, dict):
            raise OptimizationValidationError("evaluation checkpoint entries must be a mapping")
        cleaned: dict[str, dict[str, Any]] = {}
        for key, item in raw.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                raise OptimizationValidationError("invalid evaluation checkpoint entry")
            point = item.get("point")
            signed, violation = item.get("signed"), item.get("violation")
            failed = item.get("failed")
            if (
                not isinstance(point, dict)
                or any(not isinstance(k, str) or isinstance(v, bool) or
                       not isinstance(v, (float, int, str)) or
                       (isinstance(v, float) and not math.isfinite(v))
                       for k, v in point.items())
                or isinstance(signed, bool) or not isinstance(signed, (int, float))
                or not math.isfinite(signed)
                or isinstance(violation, bool) or not isinstance(violation, (int, float))
                or not math.isfinite(violation) or violation < 0
                or not isinstance(failed, bool)
            ):
                raise OptimizationValidationError("invalid numeric evaluation checkpoint entry")
            cleaned[key] = {"point": point, "signed": float(signed),
                            "violation": float(violation), "failed": failed}
        progress = data.get("progress")
        if progress is not None and not isinstance(progress, dict):
            raise OptimizationValidationError("invalid pattern search checkpoint state")
        self.entries = cleaned
        self.progress = progress

    def lookup(self, key: str) -> dict[str, Any] | None:
        value = self.entries.get(key)
        return None if value is None else dict(value)

    def record(
        self,
        key: str,
        *,
        point: Mapping[str, Any],
        signed: float,
        violation: float,
        failed: bool,
    ) -> None:
        if not isinstance(key, str) or not key:
            raise OptimizationValidationError("evaluation cache key must be nonempty")
        if not math.isfinite(signed) or not math.isfinite(violation) or violation < 0:
            raise OptimizationValidationError("evaluation cache cannot store nonfinite scores")
        entry = {"point": dict(point), "signed": float(signed),
                 "violation": float(violation), "failed": bool(failed)}
        # Serialization validation precedes mutation, so an unsupported
        # categorical value cannot corrupt an otherwise valid checkpoint.
        json.dumps(entry, allow_nan=False)
        self.entries[key] = entry
        self.save()

    def set_progress(self, state: Mapping[str, Any]) -> None:
        candidate = dict(state)
        json.dumps(candidate, allow_nan=False)
        self.progress = candidate
        self.save()

    def save(self) -> None:
        payload = {
            "version": self.SCHEMA_VERSION,
            "identity": self.identity,
            "entries": self.entries,
            "progress": self.progress,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(data.encode("utf-8")) > self.maximum_bytes:
            raise OptimizationValidationError("evaluation checkpoint exceeds configured size limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)


__all__ = ["EvaluationJournal"]
