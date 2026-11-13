"""SLAI QualityPredictor: an advisory, offline-trained batch-risk model for the Quality Agent.

The predictor estimates, *before* the full quality checks run, how likely a batch is to be blocked
or flagged, and which defect families (schema, missingness, duplicates, leakage, ...) are the likely
cause. It supplies risk tiers and a focus list. It never produces a verdict, never relaxes a policy
and never skips a mandatory check: the deterministic QualityAgent pipeline stays authoritative.

Layers
------
* ``extract_features`` - cheap, bounded, prediction-time features from a deterministic record
  sample, declared schema, provenance, the active fitness policy and the source's QualityMemory
  history. It never reads a subsystem result for the batch being scored (no target leakage).
* ``labels_from_decision`` - the one definition of "observed outcome" from a QualityAgent decision.
  Used by the trainer, by shadow-mode monitoring and by history features, so they cannot diverge.
* ``QualityPredictionModel`` - pure-Python inference of a constrained JSON checkpoint (never pickle;
  an MLP ensemble with per-target calibration, uncertainty and out-of-distribution detection).
* ``QualityPredictor`` - the QualityAgent-facing module: configuration-injected, lazily loaded,
  fail-open (an unavailable model returns ``{}`` and the gate runs exactly as before).

Modes (``quality_agent.intelligence.quality_predictor.mode`` in agents_config.yaml)
    collect   record features and the observed outcome only (training-data collection, no model)
    shadow    additionally predict and record next to the real outcome (no effect on the gate)
    advisory  additionally expose the risk tier to WorkflowControl; requires an approved checkpoint

Training lives in ``train_qp.py`` at the repository root. Inference performs no network access.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.quality_error import (
    DataQualityError,
    QualityDisposition,
    QualityDomain,
    QualityErrorType,
    QualitySeverity,
    QualityStage,
)
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]


logger = get_logger("Quality Predictor")
printer = PrettyPrinter()

MODEL_SCHEMA = 1
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = "src/agents/quality/models/quality_predictor/manifest.json"
FINAL_RECORD_TYPE = "quality_agent_final_decision"  # QualityMemory snapshot context written by QualityAgent
MAX_CHECKPOINT_BYTES = 10_000_000
MODES = ("collect", "shadow", "advisory")

USE_CASES = ("general", "knowledge_ingestion", "training", "replay", "inference_context", "memory_write")
OUTCOMES = ("block", "attention")  # attention = final verdict is warn or block
DEFECTS = (
    "schema", "required", "constraints", "missingness", "duplicates",
    "outliers", "drift", "leakage", "consistency", "provenance",
)
TARGETS = OUTCOMES + DEFECTS

# Single source of truth: defect family -> (owning subsystem, the real check names that feed it).
DEFECT_FOCUS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "schema": ("structural", ("schema_integrity",)),
    "required": ("structural", ("required_completeness",)),
    "constraints": ("structural", ("range_constraints", "domain_constraints", "coercion_health")),
    "missingness": ("statistical", ("missingness",)),
    "duplicates": ("statistical", ("duplicates",)),
    "outliers": ("statistical", ("outliers",)),
    "drift": ("statistical", ("drift",)),
    "leakage": ("semantic", ("leakage",)),
    "consistency": ("semantic", ("consistency",)),
    "provenance": ("semantic", ("provenance",)),
}
CHECK_TO_DEFECT = {check: defect for defect, (_, checks) in DEFECT_FOCUS.items() for check in checks}

FEATURES: Tuple[str, ...] = (
    # batch shape
    "log_records", "sample_fraction", "log_mean_fields", "keyset_variation",
    # completeness
    "missing_rate_mean", "missing_rate_max", "missing_field_share", "empty_record_share",
    # duplication
    "duplicate_row_rate", "duplicate_id_rate", "duplicate_text_rate",
    # types and numeric values
    "mixed_type_field_share", "type_impurity", "numeric_field_share", "outlier_rate_mean",
    "outlier_rate_max", "nonfinite_share", "constant_field_share",
    # text integrity
    "text_field_share", "log_text_length", "short_text_share", "replacement_char_rate",
    "control_char_rate", "mojibake_share",
    # declared schema
    "schema_declared", "schema_required_coverage", "schema_unknown_field_rate",
    "schema_type_violation_rate", "schema_range_violation_rate",
    # semantic proxies
    "label_present", "label_missing_rate", "label_copy_rate", "suspicious_field_name",
    "temporal_order_violation_rate", "required_if_violation_rate", "source_id_mismatch_rate",
    # provenance
    "prov_required_coverage", "prov_optional_coverage", "prov_age_known", "prov_age_log_days",
    # source history (QualityMemory)
    "history_known", "log_history_batches", "hist_block_rate", "hist_attention_rate",
    "hist_mean_score", "hist_last_score", "hist_score_trend", "hist_log_quarantine",
    "source_reliability", "reliability_known",
    *("hist_%s_rate" % d for d in DEFECTS),
    # fitness-policy context
    *("use_case_%s" % u for u in USE_CASES),
    "policy_pass_threshold", "policy_warn_threshold", "policy_requires_trusted_baseline",
    "policy_min_source_reliability", "reliability_margin", "baseline_provided",
)
FEATURE_SCHEMA_ID = hashlib.sha256("|".join((str(MODEL_SCHEMA),) + FEATURES + TARGETS).encode()).hexdigest()[:16]

_ID_FIELDS = ("id", "record_id", "row_id", "sample_id", "uuid")
_FINGERPRINT_EXCLUDE = frozenset(_ID_FIELDS) | {"created_at", "updated_at", "timestamp"}
_LEAK_BLOCK_TERMS = ("label", "target", "answer_key", "ground_truth", "expected_output", "gold_label")
_LEAK_WARN_TERMS = ("hint", "class", "predicted", "resolved_label")
_PROVENANCE_REQUIRED = ("source_id", "source_type", "collected_at")
_PROVENANCE_OPTIONAL = ("checksum", "lineage_id", "collector", "uri", "schema_version")
_MOJIBAKE = ("\u00c3", "\u00e2\u20ac", "\u00c2\u00a0", "\u00ef\u00bf\u00bd")  # typical UTF-8-read-as-Latin-1 artefacts
_TEMPORAL_PAIRS = (("start_time", "end_time"), ("created_at", "updated_at"))
_TYPE_ALIASES = {
    "str": "str", "string": "str", "text": "str", "int": "int", "integer": "int", "float": "float",
    "double": "float", "number": "float", "numeric": "float", "bool": "bool", "boolean": "bool",
    "datetime": "datetime", "date": "datetime", "timestamp": "datetime", "list": "list",
    "array": "list", "dict": "dict", "object": "dict", "mapping": "dict",
}
_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "mode": "shadow",
    "model_path": DEFAULT_MODEL_PATH,
    "allow_experimental": True,  # honoured for collect/shadow only; advisory always needs approval
    "max_sample_records": 128,
    "history_limit": 10,
    "record_features": True,
}


# =============================================================================
# Small numeric helpers
# =============================================================================
def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _ratio(numerator: float, denominator: float) -> float:
    return _clip01(numerator / denominator) if denominator > 0 else 0.0


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def _policy_value(policy: Any, key: str, default: float) -> float:
    if policy is None:
        return default
    value = policy.get(key) if isinstance(policy, Mapping) else getattr(policy, key, None)
    if isinstance(value, bool):
        return float(value)
    return _finite(value, default)


# =============================================================================
# Value classification
# =============================================================================
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, float) and math.isnan(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _value_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    return "other"


def _parse_timestamp(value: Any) -> Optional[float]:
    if _is_number(value):
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _fingerprint(record: Mapping[str, Any]) -> str:
    payload = {str(k): v for k, v in record.items() if str(k) not in _FINGERPRINT_EXCLUDE}
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)


def _sample(records: Sequence[Mapping[str, Any]], cap: int) -> List[Mapping[str, Any]]:
    """Deterministic evenly spaced sample: the same batch always yields the same features."""
    clean = [r for r in records if isinstance(r, Mapping)]
    if len(clean) <= cap:
        return clean
    step = len(clean) / cap
    return [clean[int(i * step)] for i in range(cap)]


# =============================================================================
# Feature groups (each returns only keys that are listed in FEATURES)
# =============================================================================
def _outlier_rate(values: Sequence[float]) -> float:
    """Robust outlier share (MAD rule, IQR rule when MAD collapses). Needs >= 8 observations."""
    center = median(values)
    mad = median(abs(v - center) for v in values)
    if mad > 0:
        return sum(abs(0.6745 * (v - center) / mad) > 3.5 for v in values) / len(values)
    ordered = sorted(values)
    q1, q3 = ordered[len(ordered) // 4], ordered[(3 * len(ordered)) // 4]
    iqr = q3 - q1
    if iqr <= 0:  # a mostly constant column: anything that departs from the constant is unusual
        return sum(v != center for v in values) / len(values)
    return sum(v < q1 - 3.0 * iqr or v > q3 + 3.0 * iqr for v in values) / len(values)


def _data_features(sample: Sequence[Mapping[str, Any]], total: int) -> Dict[str, float]:
    n = len(sample)
    out: Dict[str, float] = {"log_records": math.log1p(total), "sample_fraction": _ratio(n, total)}
    if n == 0:
        return out
    keysets = [frozenset(str(k) for k in r) for r in sample]
    union = sorted({k for ks in keysets for k in ks})
    out["log_mean_fields"] = math.log1p(mean(len(ks) for ks in keysets))
    out["keyset_variation"] = 1.0 - Counter(keysets).most_common(1)[0][1] / n
    if not union:
        return out
    columns = {k: [r.get(k) for r in sample] for k in union}
    missing = {k: sum(_is_missing(v) for v in col) / n for k, col in columns.items()}
    out["missing_rate_mean"] = mean(missing.values())
    out["missing_rate_max"] = max(missing.values())
    out["missing_field_share"] = sum(rate > 0.10 for rate in missing.values()) / len(union)
    out["empty_record_share"] = sum(all(_is_missing(v) for v in r.values()) for r in sample) / n

    out["duplicate_row_rate"] = 1.0 - len({_fingerprint(r) for r in sample}) / n
    ids = [next(r[k] for k in _ID_FIELDS if not _is_missing(r.get(k))) for r in sample
           if any(not _is_missing(r.get(k)) for k in _ID_FIELDS)]
    out["duplicate_id_rate"] = 1.0 - len({str(i) for i in ids}) / len(ids) if ids else 0.0

    kinds: Dict[str, Counter] = {}
    numeric: Dict[str, List[float]] = {}
    float_values = nonfinite = 0
    text_columns: Dict[str, List[str]] = {}
    for k, col in columns.items():
        present = [v for v in col if not _is_missing(v)]
        kinds[k] = Counter(_value_kind(v) for v in present)
        numbers = [v for v in present if _is_number(v)]
        float_values += sum(isinstance(v, float) for v in numbers)
        nonfinite += sum(isinstance(v, float) and not math.isfinite(v) for v in numbers)
        finite = [float(v) for v in numbers if math.isfinite(float(v))]
        if len(finite) >= 8 and k not in _ID_FIELDS:
            numeric[k] = finite
        texts = [v for v in present if isinstance(v, str)]
        if texts and len(texts) >= 0.5 * len(present) and mean(len(t) for t in texts) >= 20:
            text_columns[k] = texts
    typed = {k: c for k, c in kinds.items() if sum(c.values()) > 0}
    out["mixed_type_field_share"] = sum(len(c) > 1 for c in typed.values()) / len(union)
    out["type_impurity"] = mean(1.0 - max(c.values()) / sum(c.values()) for c in typed.values()) if typed else 0.0
    out["numeric_field_share"] = len(numeric) / len(union)
    rates = [_outlier_rate(v) for v in numeric.values()]
    out["outlier_rate_mean"] = mean(rates) if rates else 0.0
    out["outlier_rate_max"] = max(rates) if rates else 0.0
    out["nonfinite_share"] = _ratio(nonfinite, float_values)
    out["constant_field_share"] = sum(
        1 for k, col in columns.items()
        if k not in _ID_FIELDS and n >= 8 and len({json.dumps(v, default=str) for v in col if not _is_missing(v)}) <= 1
    ) / len(union)

    out["text_field_share"] = len(text_columns) / len(union)
    if text_columns:
        values = [t for col in text_columns.values() for t in col]
        chars = max(sum(len(t) for t in values), 1)
        out["log_text_length"] = math.log1p(mean(len(t) for t in values))
        out["short_text_share"] = sum(len(t.strip()) < 3 for t in values) / len(values)
        out["replacement_char_rate"] = _clip01(sum(t.count("\ufffd") for t in values) / chars)
        out["control_char_rate"] = _clip01(
            sum(1 for t in values for c in t if ord(c) < 32 and c not in "\n\r\t") / chars)
        out["mojibake_share"] = sum(any(m in t for m in _MOJIBAKE) for t in values) / len(values)
        longest = max(text_columns, key=lambda k: mean(len(t) for t in text_columns[k]))
        column = text_columns[longest]
        out["duplicate_text_rate"] = 1.0 - len(set(column)) / len(column)
    return out


def _schema_specs(schema: Any) -> Tuple[Dict[str, Dict[str, Any]], set]:
    """Field specs and required names from the schema shapes StructuralQuality accepts."""
    if not isinstance(schema, Mapping):
        return {}, set()
    as_map = lambda value: value if isinstance(value, Mapping) else {}  # noqa: E731
    raw_required = schema.get("required_fields")
    if raw_required is None:
        raw_required = schema.get("required")
    if isinstance(raw_required, Mapping):
        required = {str(k) for k, enabled in raw_required.items() if enabled}
    elif isinstance(raw_required, (list, tuple, set)):
        required = {str(f) for f in raw_required}
    else:
        required = set()
    merged: Dict[str, Any] = {str(k): v for k, v in as_map(schema.get("fields")).items()}
    for name, kind in as_map(schema.get("field_types")).items():
        merged.setdefault(str(name), {"type": kind})
    for name, constraint in as_map(schema.get("constraints")).items():
        current = merged.get(str(name))
        base = {"type": current} if isinstance(current, str) else dict(as_map(current))
        base.update(as_map(constraint))
        merged[str(name)] = base
    specs: Dict[str, Dict[str, Any]] = {}
    for name, definition in merged.items():
        definition = {"type": definition} if isinstance(definition, str) else as_map(definition)
        if definition.get("required"):
            required.add(name)
        enum = definition.get("enum", definition.get("allowed_values"))
        specs[name] = {
            "type": _TYPE_ALIASES.get(str(definition.get("type", "any")).strip().lower(), "any"),
            "min": definition.get("min"), "max": definition.get("max"),
            "enum": {str(e) for e in enum} if isinstance(enum, (list, tuple, set)) else None,
        }
    for name in required:
        specs.setdefault(name, {"type": "any", "min": None, "max": None, "enum": None})
    return specs, required


def _type_matches(value: Any, kind: str) -> bool:
    if kind == "str":
        return isinstance(value, str)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return _is_number(value)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "datetime":
        return _parse_timestamp(value) is not None
    if kind == "list":
        return isinstance(value, (list, tuple))
    if kind == "dict":
        return isinstance(value, Mapping)
    return True


def _schema_features(sample: Sequence[Mapping[str, Any]], schema: Any) -> Dict[str, float]:
    specs, required = _schema_specs(schema)
    out: Dict[str, float] = {"schema_declared": float(bool(specs)), "schema_required_coverage": 1.0}
    if not specs or not sample:
        return out
    if required:
        present = sum(not _is_missing(r.get(f)) for r in sample for f in required)
        out["schema_required_coverage"] = present / (len(sample) * len(required))
    out["schema_unknown_field_rate"] = mean(
        sum(str(k) not in specs for k in r) / max(len(r), 1) for r in sample)
    typed = type_bad = ranged = range_bad = 0
    for record in sample:
        for name, spec in specs.items():
            value = record.get(name)
            if _is_missing(value):
                continue
            if spec["type"] != "any":
                typed += 1
                type_bad += not _type_matches(value, spec["type"])
            if _is_number(value) and (spec["min"] is not None or spec["max"] is not None):
                ranged += 1
                low, high = _finite(spec["min"], -math.inf), _finite(spec["max"], math.inf)
                range_bad += not (low <= float(value) <= high)
            elif spec["enum"] is not None:
                ranged += 1
                range_bad += str(value) not in spec["enum"]
    out["schema_type_violation_rate"] = _ratio(type_bad, typed)
    out["schema_range_violation_rate"] = _ratio(range_bad, ranged)
    return out


def _semantic_features(sample: Sequence[Mapping[str, Any]], source_id: Optional[str],
                       label_field: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not sample:
        return out
    label = (label_field or "label").strip() or "label"
    fields = sorted({str(k) for r in sample for k in r})
    n = len(sample)
    out["label_present"] = float(label in fields)
    if label in fields:
        out["label_missing_rate"] = sum(_is_missing(r.get(label)) for r in sample) / n
        copies = []
        for f in fields:
            if f == label or f in _ID_FIELDS:
                continue
            both = [(r.get(f), r.get(label)) for r in sample
                    if not _is_missing(r.get(f)) and not _is_missing(r.get(label))]
            if both:
                copies.append(sum(str(a).strip().lower() == str(b).strip().lower() for a, b in both) / len(both))
        out["label_copy_rate"] = max(copies) if copies else 0.0
    lowered = [f.lower() for f in fields if f != label]
    out["suspicious_field_name"] = (
        1.0 if any(t in f for f in lowered for t in _LEAK_BLOCK_TERMS)
        else 0.5 if any(t in f for f in lowered for t in _LEAK_WARN_TERMS) else 0.0)

    violations = eligible = 0
    for early, late in _TEMPORAL_PAIRS:
        for r in sample:
            a, b = _parse_timestamp(r.get(early)), _parse_timestamp(r.get(late))
            if a is not None and b is not None:
                eligible += 1
                violations += a > b
    out["temporal_order_violation_rate"] = _ratio(violations, eligible)
    resolved = [r for r in sample if str(r.get("status", "")).strip().lower() == "resolved"]
    out["required_if_violation_rate"] = _ratio(sum(_is_missing(r.get("resolved_at")) for r in resolved), len(resolved))
    claimed = [r for r in sample if not _is_missing(r.get("source_id"))]
    out["source_id_mismatch_rate"] = (
        _ratio(sum(str(r["source_id"]) != str(source_id) for r in claimed), len(claimed)) if source_id else 0.0)
    return out


def _provenance_features(source_id: Optional[str], provenance: Any, source_metadata: Any,
                         now: float) -> Dict[str, float]:
    merged: Dict[str, Any] = {}
    for part in (source_metadata, provenance):
        if isinstance(part, Mapping):
            merged.update({str(k): v for k, v in part.items()})
    if source_id:
        merged.setdefault("source_id", source_id)
    present = lambda key: float(not _is_missing(merged.get(key)))  # noqa: E731
    out = {
        "prov_required_coverage": mean(present(k) for k in _PROVENANCE_REQUIRED),
        "prov_optional_coverage": mean(present(k) for k in _PROVENANCE_OPTIONAL),
    }
    collected = _parse_timestamp(merged.get("collected_at"))
    if collected is not None:
        out["prov_age_known"] = 1.0
        out["prov_age_log_days"] = math.log1p(max(0.0, (now - collected) / 86400.0))
    return out


def _history_features(memory: Any, source_id: Optional[str], limit: int) -> Dict[str, float]:
    out = {"source_reliability": 0.5}
    if memory is None or not source_id:
        return out
    try:
        snapshots = memory.quality_snapshots(source_id, limit=max(limit * 8, 32))
        reliability = memory.latest_source_reliability(source_id)
    except Exception as exc:  # history is optional evidence, never a reason to fail the gate
        logger.debug("QualityPredictor history unavailable (%s)", type(exc).__name__)
        return out
    if isinstance(reliability, Mapping) and reliability.get("reliability") is not None:
        out["source_reliability"] = _clip01(_finite(reliability["reliability"], 0.5))
        out["reliability_known"] = 1.0
    finals = sorted(
        (s for s in snapshots or [] if isinstance(s, Mapping)
         and (s.get("context") or {}).get("record_type") == FINAL_RECORD_TYPE),
        key=lambda s: _finite(s.get("created_at")))[-limit:]
    if not finals:
        return out
    scores = [_clip01(_finite(s.get("batch_score"), 0.0)) for s in finals]
    verdicts = [str(s.get("verdict", "warn")).lower() for s in finals]
    observed = [labels_from_decision({"verdict": v, "subsystem_findings": s.get("checker_findings")})
                for s, v in zip(finals, verdicts)]
    out.update({
        "history_known": 1.0,
        "log_history_batches": math.log1p(len(finals)),
        "hist_block_rate": mean(v == "block" for v in verdicts),
        "hist_attention_rate": mean(v != "pass" for v in verdicts),
        "hist_mean_score": mean(scores),
        "hist_last_score": scores[-1],
        "hist_log_quarantine": math.log1p(mean(max(0.0, _finite(s.get("quarantine_count"))) for s in finals)),
    })
    if len(scores) >= 4:
        out["hist_score_trend"] = max(-1.0, min(1.0, mean(scores[-3:]) - mean(scores[:-3])))
    for defect in DEFECTS:
        seen = [o[defect] for o in observed if o[defect] is not None]
        out["hist_%s_rate" % defect] = mean(seen) if seen else 0.0
    return out


def extract_features(
    records: Sequence[Mapping[str, Any]],
    *,
    source_id: Optional[str] = None,
    use_case: str = "general",
    policy: Any = None,
    schema: Optional[Mapping[str, Any]] = None,
    label_field: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    source_metadata: Optional[Mapping[str, Any]] = None,
    baseline_provided: bool = False,
    memory: Any = None,
    max_sample_records: int = _DEFAULTS["max_sample_records"],
    history_limit: int = _DEFAULTS["history_limit"],
    now: Optional[float] = None,
) -> Dict[str, float]:
    """Prediction-time features for one batch. Every key of ``FEATURES`` is present and finite."""
    clock = time.time() if now is None else float(now)
    sample = _sample(records or [], max(1, int(max_sample_records)))
    raw: Dict[str, float] = {}
    raw.update(_data_features(sample, len(records or [])))
    raw.update(_schema_features(sample, schema))
    raw.update(_semantic_features(sample, source_id, label_field))
    raw.update(_provenance_features(source_id, provenance, source_metadata, clock))
    raw.update(_history_features(memory, source_id, max(1, int(history_limit))))
    for case in USE_CASES:
        raw["use_case_%s" % case] = float(case == use_case)
    reliability = raw["source_reliability"]
    min_reliability = _policy_value(policy, "min_source_reliability", 0.5)
    raw.update({
        "policy_pass_threshold": _policy_value(policy, "pass_threshold", 0.90),
        "policy_warn_threshold": _policy_value(policy, "warn_threshold", 0.75),
        "policy_requires_trusted_baseline": _policy_value(policy, "require_trusted_baseline", 0.0),
        "policy_min_source_reliability": min_reliability,
        "reliability_margin": max(-1.0, min(1.0, reliability - min_reliability)),
        "baseline_provided": float(bool(baseline_provided)),
    })
    unknown = set(raw) - set(FEATURES)
    if unknown:  # a programming error, not a data problem
        raise RuntimeError("QualityPredictor produced unregistered features: " + ", ".join(sorted(unknown)))
    return {name: _finite(raw.get(name, 0.0)) for name in FEATURES}


# =============================================================================
# Observed outcome (the single definition used by trainer, monitoring and history)
# =============================================================================
def labels_from_decision(decision: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Outcome targets from a QualityAgent decision.

    ``block``/``attention`` come from the final verdict. A defect family is 1.0 when any of its real
    checks returned a non-pass verdict, 0.0 when the checks ran and passed, and ``None`` when no
    check of that family ran (for example after a structural block skipped later subsystems).
    """
    verdict = str(decision.get("verdict", "warn")).strip().lower()
    out: Dict[str, Optional[float]] = {"block": float(verdict == "block"), "attention": float(verdict != "pass")}
    seen: Dict[str, bool] = {}
    for finding in decision.get("subsystem_findings") or []:
        if not isinstance(finding, Mapping):
            continue
        defect = CHECK_TO_DEFECT.get(str(finding.get("check_name") or finding.get("check") or ""))
        if defect is not None:
            failed = str(finding.get("verdict", "pass")).strip().lower() != "pass"
            seen[defect] = seen.get(defect, False) or failed
    out.update({d: (float(seen[d]) if d in seen else None) for d in DEFECTS})
    return out


# =============================================================================
# Model: constrained JSON checkpoint, pure-Python inference
# =============================================================================
class QualityPredictionModel:
    """Immutable inference over a validated checkpoint (MLP ensemble + calibration)."""

    def __init__(self, manifest_path: str | Path, *, allow_experimental: bool = False) -> None:
        path = Path(manifest_path)
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("QualityPredictor checkpoint exceeds the size limit")
        raw = path.read_bytes()
        self._init(json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest(), allow_experimental)

    @classmethod
    def from_model(cls, model: Mapping[str, Any], *, allow_experimental: bool = True) -> "QualityPredictionModel":
        """Build from an in-memory checkpoint (trainer parity checks, tests)."""
        obj = cls.__new__(cls)
        raw = json.dumps(model, sort_keys=True, allow_nan=False).encode("utf-8")
        obj._init(json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest(), allow_experimental)
        return obj

    def _init(self, model: Dict[str, Any], checksum: str, allow_experimental: bool) -> None:
        if model.get("schema") != MODEL_SCHEMA:
            raise ValueError("Incompatible QualityPredictor schema")
        if tuple(model.get("features", ())) != FEATURES or tuple(model.get("targets", ())) != TARGETS:
            raise ValueError("Incompatible QualityPredictor feature or target order")
        promotion = model.get("promotion")
        if promotion not in ("experimental", "approved"):
            raise ValueError("Unknown QualityPredictor promotion state")
        if promotion != "approved" and not allow_experimental:
            raise ValueError("QualityPredictor checkpoint is not production-approved")
        n_features, n_targets = len(FEATURES), len(TARGETS)
        scaler, domain = model.get("scaler", {}), model.get("domain", {})
        for block, keys in ((scaler, ("mean", "scale")), (domain, ("min", "max"))):
            for key in keys:
                if len(block.get(key, ())) != n_features or not all(_is_finite_number(v) for v in block[key]):
                    raise ValueError("Invalid QualityPredictor %s vector" % key)
        if any(v <= 0 for v in scaler["scale"]):
            raise ValueError("QualityPredictor scaler has non-positive scales")
        members = model.get("members") or []
        if not members:
            raise ValueError("QualityPredictor checkpoint has no ensemble members")
        for member in members:
            w1, b1, w2, b2 = member.get("W1"), member.get("b1"), member.get("W2"), member.get("b2")
            hidden = len(b1 or ())
            if (not hidden or len(w1 or ()) != n_features or any(len(row) != hidden for row in w1)
                    or len(w2 or ()) != hidden or any(len(row) != n_targets for row in w2) or len(b2 or ()) != n_targets):
                raise ValueError("QualityPredictor member dimensions do not match the schema")
            flat = [*b1, *b2, *(v for row in w1 for v in row), *(v for row in w2 for v in row)]
            if not all(_is_finite_number(v) for v in flat):
                raise ValueError("Invalid QualityPredictor network values")
        calibration = model.get("calibration") or []
        if len(calibration) != n_targets or not all(len(p) == 2 and all(_is_finite_number(v) for v in p) for p in calibration):
            raise ValueError("Invalid QualityPredictor calibration")
        selection = model.get("selection") or {}
        for key in ("low_risk_below", "high_risk_at", "focus_threshold", "max_uncertainty", "max_ood_fraction"):
            if not _is_finite_number(selection.get(key)) or not 0.0 <= selection[key] <= 1.0:
                raise ValueError("Invalid QualityPredictor selection value: " + key)
        if selection["low_risk_below"] > selection["high_risk_at"]:
            raise ValueError("QualityPredictor low-risk threshold exceeds the high-risk threshold")
        self.model = model
        self.checksum = checksum
        self.version = str(model.get("version", "unknown"))
        self.promotion = str(promotion)
        self.selection = dict(selection)

    # -- numerics ---------------------------------------------------------------------------------
    def _standardize(self, features: Mapping[str, float]) -> List[float]:
        scaler = self.model["scaler"]
        x = []
        for name, mu, sc in zip(FEATURES, scaler["mean"], scaler["scale"]):
            value = float(features.get(name, 0.0))
            if not math.isfinite(value):
                raise ValueError("Non-finite QualityPredictor input feature: " + name)
            x.append(max(-10.0, min(10.0, (value - mu) / sc)))
        return x

    @staticmethod
    def _forward(member: Mapping[str, Any], x: Sequence[float]) -> List[float]:
        w1, b1, w2, b2 = member["W1"], member["b1"], member["W2"], member["b2"]
        hidden = [max(0.0, b + sum(xi * row[j] for xi, row in zip(x, w1) if xi)) for j, b in enumerate(b1)]
        return [b2[t] + sum(h * row[t] for h, row in zip(hidden, w2) if h) for t in range(len(TARGETS))]

    def ood_fraction(self, features: Mapping[str, float]) -> float:
        lows, highs = self.model["domain"]["min"], self.model["domain"]["max"]
        outside = 0
        for name, lo, hi in zip(FEATURES, lows, highs):
            margin = 0.1 * (hi - lo) + 1e-9
            value = float(features.get(name, 0.0))
            outside += value < lo - margin or value > hi + margin
        return outside / len(FEATURES)

    def score(self, features: Mapping[str, float]) -> Dict[str, Any]:
        """Calibrated target probabilities, ensemble disagreement and out-of-distribution share."""
        x = self._standardize(features)
        logits = [self._forward(member, x) for member in self.model["members"]]
        count = len(logits)
        mean_logit = [sum(member[t] for member in logits) / count for t in range(len(TARGETS))]
        probabilities = {
            target: _sigmoid(a * z + b)
            for target, z, (a, b) in zip(TARGETS, mean_logit, self.model["calibration"])
        }
        spread = 0.0
        if count > 1:
            for target in OUTCOMES:
                t = TARGETS.index(target)
                member_probs = [_sigmoid(member[t]) for member in logits]
                centre = sum(member_probs) / count
                spread += math.sqrt(sum((p - centre) ** 2 for p in member_probs) / count)
            spread /= len(OUTCOMES)
        return {"probabilities": probabilities, "uncertainty": spread, "ood_fraction": self.ood_fraction(features)}

    def assess(self, features: Mapping[str, float]) -> Dict[str, Any]:
        """Risk tier and focus list. Advisory only: no verdict, no policy change."""
        scored = self.score(features)
        probs, sel = scored["probabilities"], self.selection
        p_block, p_attention = probs["block"], probs["attention"]
        if p_block >= sel["high_risk_at"]:
            tier = "high"
        elif max(p_block, p_attention) < sel["low_risk_below"]:
            tier = "low"
        else:
            tier = "elevated"
        reliable = scored["uncertainty"] <= sel["max_uncertainty"] and scored["ood_fraction"] <= sel["max_ood_fraction"]
        focus = [
            {"defect": d, "probability": round(probs[d], 6), "subsystem": DEFECT_FOCUS[d][0], "checks": list(DEFECT_FOCUS[d][1])}
            for d in sorted(DEFECTS, key=lambda d: -probs[d]) if probs[d] >= sel["focus_threshold"]
        ]
        return {
            "model_version": self.version,
            "model_sha256": self.checksum,
            "promotion": self.promotion,
            "feature_schema": FEATURE_SCHEMA_ID,
            "probabilities": {t: round(p, 6) for t, p in probs.items()},
            "block_probability": round(p_block, 6),
            "attention_probability": round(p_attention, 6),
            "uncertainty": round(scored["uncertainty"], 6),
            "ood_fraction": round(scored["ood_fraction"], 6),
            "reliable": bool(reliable),
            "risk_tier": tier,
            "advice": {
                "focus": focus,
                # Only an approved checkpoint, confident and in-distribution, may suggest trimming
                # *optional* analysis. Mandatory checks and policy requirements are never affected.
                "deprioritize_optional_checks": bool(tier == "low" and reliable and self.promotion == "approved"),
                "mandatory_checks_unchanged": True,
            },
        }


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# =============================================================================
# Quality Agent module
# =============================================================================
class QualityPredictor:
    """Configuration-injected, fail-open predictor owned by the QualityAgent."""

    def __init__(self, *, memory: Any = None, config: Optional[Mapping[str, Any]] = None) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise self._config_error("quality_predictor config must be a mapping")
        section = dict(_DEFAULTS)
        section.update(dict(config or {}))
        self.config = section
        self.memory = memory
        self.enabled = bool(section["enabled"])
        self.mode = str(section["mode"]).strip().lower()
        if self.mode not in MODES:
            raise self._config_error("quality_predictor.mode must be one of: " + ", ".join(MODES))
        self.max_sample_records = self._positive(section["max_sample_records"], "max_sample_records")
        self.history_limit = self._positive(section["history_limit"], "history_limit")
        self.record_features = bool(section["record_features"]) or self.mode == "collect"
        # Advisory use of an unapproved model is never allowed.
        self.allow_experimental = bool(section["allow_experimental"]) and self.mode != "advisory"
        path = Path(str(section["model_path"]))
        self.model_path = path if path.is_absolute() else PROJECT_ROOT / path
        self._model: Optional[QualityPredictionModel] = None
        self._load_failed = False
        self._lock = threading.RLock()
        if self.enabled:
            printer.status("INIT", "Initializing Quality Predictor (%s)" % self.mode, "info")

    # -- configuration helpers --------------------------------------------------------------------
    @staticmethod
    def _config_error(message: str) -> DataQualityError:
        return DataQualityError(
            message=message,
            error_type=QualityErrorType.CONFIGURATION_INVALID,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.VALIDATION,
            domain=QualityDomain.SYSTEM,
            disposition=QualityDisposition.ESCALATE,
            remediation="Fix quality_agent.intelligence.quality_predictor in agents_config.yaml.",
        )

    def _positive(self, value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise self._config_error("quality_predictor.%s must be a positive integer" % name)
        return value

    # -- model ------------------------------------------------------------------------------------
    def _get_model(self) -> Optional[QualityPredictionModel]:
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is None and not self._load_failed:
                try:
                    self._model = QualityPredictionModel(self.model_path, allow_experimental=self.allow_experimental)
                    logger.info("QualityPredictor loaded | version=%s | promotion=%s", self._model.version, self._model.promotion)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.warning("QualityPredictor unavailable; deterministic gate retained (%s: %s)", type(exc).__name__, exc)
                    self._load_failed = True
            return self._model

    # -- public API -------------------------------------------------------------------------------
    def predict_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_id: Optional[str],
        use_case: str = "general",
        policy: Any = None,
        schema: Optional[Mapping[str, Any]] = None,
        label_field: Optional[str] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Any]] = None,
        baseline_provided: bool = False,
    ) -> Dict[str, Any]:
        """Pre-check prediction for one batch; ``{}`` when disabled or on any failure."""
        if not self.enabled:
            return {}
        try:
            features = extract_features(
                records, source_id=source_id, use_case=use_case, policy=policy, schema=schema,
                label_field=label_field, provenance=provenance, source_metadata=source_metadata,
                baseline_provided=baseline_provided, memory=self.memory,
                max_sample_records=self.max_sample_records, history_limit=self.history_limit)
            result: Dict[str, Any] = {"mode": self.mode, "authority": "advisory_only", "feature_schema": FEATURE_SCHEMA_ID}
            if self.record_features:
                result["features"] = {k: round(v, 6) for k, v in features.items()}
            if self.mode == "collect":
                return result
            model = self._get_model()
            if model is None:
                result["status"] = "model_unavailable"
                return result
            result.update(model.assess(features))
            return result
        except Exception as exc:  # noqa: BLE001 - advisory component must never break the gate
            logger.warning("QualityPredictor inference failed; deterministic gate retained (%s: %s)", type(exc).__name__, exc)
            return {}

    @staticmethod
    def observe(prediction: Mapping[str, Any], decision: Mapping[str, Any]) -> Dict[str, Any]:
        """Attach the real outcome to a prediction so shadow-mode quality and retraining data accrue."""
        if not prediction:
            return {}
        observed = labels_from_decision(decision)
        out: Dict[str, Any] = {"observed": observed}
        probabilities = prediction.get("probabilities")
        if isinstance(probabilities, Mapping):
            out["absolute_error"] = {
                t: round(abs(float(probabilities[t]) - y), 6)
                for t, y in observed.items() if y is not None and t in probabilities}
        return out

    def status(self) -> Dict[str, Any]:
        model = self._model
        return {
            "enabled": self.enabled, "mode": self.mode, "model_path": str(self.model_path),
            "model_loaded": model is not None, "model_load_failed": self._load_failed,
            "model_version": model.version if model else None, "promotion": model.promotion if model else None,
            "feature_schema": FEATURE_SCHEMA_ID,
        }


__all__ = ["QualityPredictionModel", "QualityPredictor"]
