"""SLAI ReasoningStrategySelector (RSS): offline, read-only reasoning-style ranking.

Scores the reasoning styles that ``ReasoningTypes`` can execute and returns a selection
in the same shape as ``ReasoningTypes.select_reasoning_strategy`` (``strategy``,
``method``, ``matches``), so the two are drop-in interchangeable.

RSS only *chooses*; it never executes a strategy, never touches the network, and never
changes ReasoningTypes. The checkpoint is constrained JSON (no pickle) and inference is
pure Python, so it runs anywhere SLAI runs. Training lives in ``train_rss.py``; feature
extraction deliberately lives here so training and runtime cannot drift apart.

Selection utility, for candidate style ``s`` and task ``x`` (mirrors the design note):

    U(s|x) = P(s is appropriate / yields a validated result | x) - mu * T(x, s)

where ``T`` is predicted latency, normalised to [0, 1] by ``latency_cap_ms``. The latency
term is only active when the checkpoint was trained on observed latencies; otherwise
``mu * T`` is 0 and ``U`` is the calibrated probability.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Strategy Selector")
printer = PrettyPrinter()


MODEL_SCHEMA = "slai.reasoning.strategy_selector.v1"
# Canonical names: identical to reasoning_config.yaml (reasoning_types.strategy_keywords).
STRATEGIES = ("abduction", "deduction", "induction", "analogical", "decompositional", "cause_effect")
# Structured inputs ReasoningTypes._invoke_component reads from context or a mapping problem.
CONTEXT_KEYS = ("premises", "hypothesis", "observations", "events", "conditions",
                "source_domain", "target", "system")
COUNTED_KEYS = ("premises", "observations", "events")
TEXT_KEYS = ("problem", "question", "goal", "query", "text", "prompt")
DENSE_FEATURES = (
    "log_chars", "log_tokens", "sentences", "question_marks", "digit_ratio",
    "number_tokens", "comma_ratio", "list_markers", "upper_ratio",
    *("has_" + k for k in CONTEXT_KEYS),
    *("count_" + k for k in COUNTED_KEYS),
    *("kw_" + s for s in STRATEGIES),
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "strategy_selector" / "model.json"
MAX_ARTIFACT_BYTES = 16_000_000
MAX_TEXT_CHARS = 20_000

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
_LIST_MARKER_RE = re.compile(r"\s*(?:[-*]|\d+[.)])\s+")


# ----------------------------------------------------------------------------
# Feature extraction (shared by train_rss.py and runtime inference)
# ----------------------------------------------------------------------------
def problem_text(problem: Any) -> str:
    """Plain text of a ReasoningAgent problem (str, mapping payload, or sequence)."""
    if problem is None:
        return ""
    if isinstance(problem, str):
        text = problem
    elif isinstance(problem, Mapping):
        parts = [problem[k] for k in TEXT_KEYS if isinstance(problem.get(k), str)]
        text = " ".join(parts or [v for v in problem.values() if isinstance(v, str)])
    elif isinstance(problem, (list, tuple)):
        text = " ".join(str(v) for v in problem if isinstance(v, (str, int, float)))
    else:
        text = str(problem)
    return text[:MAX_TEXT_CHARS]


def text_grams(problem: Any) -> list[str]:
    """Unigrams and bigrams of the problem text (the vocabulary the checkpoint stores)."""
    tokens = _TOKEN_RE.findall(problem_text(problem).lower())
    return ["u:" + t for t in tokens] + ["b:" + a + " " + b for a, b in zip(tokens, tokens[1:])]


def _present(value: Any) -> bool:
    if value is None:
        return False
    return len(value) > 0 if hasattr(value, "__len__") else True


def _count(value: Any) -> int:
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 1 if _present(value) else 0


def featurize(problem: Any, context: Optional[Mapping[str, Any]] = None, *, vocab: Mapping[str, int],
              keywords: Optional[Mapping[str, Sequence[str]]] = None) -> dict[int, float]:
    """Sparse feature vector: vocabulary n-grams (L2-normalised) + dense shape/context/keyword hits.

    Indices ``[0, len(vocab))`` are n-grams; ``len(vocab) + i`` is ``DENSE_FEATURES[i]``. Grams
    outside the checkpoint's vocabulary are ignored, so unseen wording cannot collide with
    learned wording. All dense features are scaled into [0, 1].
    """
    text = problem_text(problem)
    lowered = text.lower()
    counts = Counter(g for g in text_grams(text) if g in vocab)
    sparse = {vocab[g]: math.log1p(c) for g, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in sparse.values()))
    out = {i: v / norm for i, v in sparse.items()} if norm > 0 else {}

    n_chars, n_tokens = len(text), len(_TOKEN_RE.findall(lowered))
    letters = sum(c.isalpha() for c in text)
    dense = {
        "log_chars": math.log1p(n_chars) / math.log1p(2000),
        "log_tokens": math.log1p(n_tokens) / math.log1p(400),
        "sentences": len([s for s in _SENTENCE_RE.split(text) if s.strip()]) / 8.0,
        "question_marks": text.count("?") / 3.0,
        "digit_ratio": 5.0 * sum(c.isdigit() for c in text) / max(n_chars, 1),
        "number_tokens": len(_NUMBER_RE.findall(text)) / 20.0,
        "comma_ratio": 5.0 * text.count(",") / max(n_tokens + 1, 1),
        "list_markers": sum(bool(_LIST_MARKER_RE.match(line)) for line in text.splitlines()) / 6.0,
        "upper_ratio": 5.0 * sum(c.isupper() for c in text) / max(letters, 1),
    }
    ctx = context if isinstance(context, Mapping) else {}
    payload = problem if isinstance(problem, Mapping) else {}
    for key in CONTEXT_KEYS:
        value = ctx.get(key)
        if value is None:
            value = payload.get(key)
        dense["has_" + key] = 1.0 if _present(value) else 0.0
        if key in COUNTED_KEYS:
            dense["count_" + key] = math.log1p(_count(value)) / math.log1p(50)
    for name in STRATEGIES:  # the existing keyword policy's evidence, as a learnable (not binding) hint
        hits = sum(str(w).lower() in lowered for w in (keywords or {}).get(name, ()))
        dense["kw_" + name] = hits / 2.0
    base = len(vocab)
    for offset, name in enumerate(DENSE_FEATURES):
        v = min(1.0, max(0.0, dense.get(name, 0.0)))
        if v > 0.0:
            out[base + offset] = v
    return out


# ----------------------------------------------------------------------------
# Selection rule (shared by the runtime and by train_rss.py threshold tuning)
# ----------------------------------------------------------------------------
def strategy_utility(probability: float, expected_log_ms: Optional[float],
                     latency_weight: float, latency_cap_ms: float) -> float:
    """U = P - mu * T, with T = predicted log1p(ms) normalised by log1p(cap), clipped to [0, 1]."""
    if expected_log_ms is None or latency_weight <= 0.0:
        return probability
    t = min(1.0, max(0.0, expected_log_ms / math.log1p(latency_cap_ms)))
    return probability - latency_weight * t


def choose_strategies(utilities: Mapping[str, float], max_strategies: int, threshold: float) -> list[str]:
    """Best style always; further styles only while ``U >= threshold`` and under the cap."""
    ranked = sorted(utilities.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = [ranked[0][0]]
    for name, utility in ranked[1:]:
        if len(chosen) >= max(1, max_strategies):
            break
        if utility >= threshold:
            chosen.append(name)
    return chosen


def order_sequence(chosen: Sequence[str], precedence: Sequence[Sequence[float]]) -> list[str]:
    """Order styles by learned pairwise precedence ``precedence[a][b] = P(a runs before b)``."""
    if len(chosen) < 2:
        return list(chosen)
    index = {name: i for i, name in enumerate(STRATEGIES)}

    def total(perm: Sequence[str]) -> float:
        return sum(math.log(precedence[index[a]][index[b]]) for a, b in itertools.combinations(perm, 2))

    return list(max(itertools.permutations(chosen), key=total))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def _check_matrix(value: Any, rows: int, cols: int, name: str) -> None:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"RSS {name} must have {rows} rows")
    for row in value:
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"RSS {name} rows must have {cols} columns")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in row):
            raise ValueError(f"RSS {name} contains non-finite or non-numeric values")


def _check_vector(value: Any, size: int, name: str) -> None:
    _check_matrix([value], 1, size, name)


# ----------------------------------------------------------------------------
# Runtime model
# ----------------------------------------------------------------------------
class ReasoningStrategySelector:
    """Immutable numerical inference from a validated, locally stored RSS checkpoint."""

    def __init__(self, manifest_path: str | Path, *, allow_experimental: bool = False) -> None:
        path = Path(manifest_path)
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("RSS artifact exceeds size limit")
        raw = path.read_bytes()
        self._init(json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest(), allow_experimental)

        logger.info(f"RSS successfully initialized with {raw}.")

    @classmethod
    def from_model(cls, model: Mapping[str, Any], *, allow_experimental: bool = True) -> "ReasoningStrategySelector":
        """Build from an in-memory checkpoint dict (used by the trainer before writing to disk)."""
        canonical = json.dumps(model, sort_keys=True, allow_nan=False).encode("utf-8")
        self = cls.__new__(cls)
        self._init(json.loads(canonical.decode("utf-8")), hashlib.sha256(canonical).hexdigest(), allow_experimental)
        return self

    @classmethod
    def try_load(cls, path: str | Path | None = None, *, allow_experimental: bool = False
                 ) -> Optional["ReasoningStrategySelector"]:
        """Load the default checkpoint, or return None when none is installed.

        A present-but-invalid checkpoint still raises: corruption must not silently
        degrade to the keyword policy.
        """
        target = Path(path) if path else DEFAULT_MODEL_PATH
        return cls(target, allow_experimental=allow_experimental) if target.is_file() else None

    def _init(self, model: dict, checksum: str, allow_experimental: bool) -> None:
        if model.get("schema") != MODEL_SCHEMA:
            raise ValueError("Incompatible RSS schema")
        if tuple(model.get("strategies", ())) != STRATEGIES:
            raise ValueError("RSS strategy set does not match this SLAI build")
        if not allow_experimental and model.get("promotion") != "approved":
            raise ValueError("RSS checkpoint is not production-approved")
        feats = model.get("features", {})
        vocab_list = feats.get("vocab")
        if (not isinstance(vocab_list, list) or not 1 <= len(vocab_list) <= 200_000
                or not all(isinstance(g, str) for g in vocab_list) or len(set(vocab_list)) != len(vocab_list)):
            raise ValueError("RSS vocabulary is invalid")
        if tuple(feats.get("dense", ())) != DENSE_FEATURES:
            raise ValueError("Incompatible RSS dense feature schema")
        keywords = feats.get("keywords", {})
        if not isinstance(keywords, dict) or not all(
                k in STRATEGIES and isinstance(v, list) and all(isinstance(w, str) for w in v) for k, v in keywords.items()):
            raise ValueError("RSS keyword snapshot is invalid")
        k, d = len(STRATEGIES), len(vocab_list) + len(DENSE_FEATURES)
        net = model.get("network", {})
        hidden = net.get("b1") and len(net["b1"])
        if not hidden:
            raise ValueError("RSS network is missing hidden units")
        _check_matrix(net.get("W1"), d, hidden, "network.W1")
        _check_vector(net["b1"], hidden, "network.b1")
        heads = model.get("heads", {})
        prob = heads.get("probability", {})
        _check_matrix(prob.get("W"), k, hidden, "heads.probability.W")
        _check_vector(prob.get("b"), k, "heads.probability.b")
        _check_matrix(prob.get("calibration"), k, 2, "heads.probability.calibration")
        if any(a <= 0 for a, _ in prob["calibration"]):
            raise ValueError("RSS calibration slopes must be positive")
        lat = heads.get("latency")
        if lat is not None:
            _check_matrix(lat.get("W"), k, hidden, "heads.latency.W")
            _check_vector(lat.get("b"), k, "heads.latency.b")
        sel = model.get("selection", {})
        if not (0.0 <= sel.get("threshold", -1) <= 1.0 and 0.0 <= sel.get("min_confidence", -1) <= 1.0
                and sel.get("latency_weight", -1) >= 0.0 and sel.get("latency_cap_ms", 0) > 0
                and 1 <= sel.get("max_strategies", 0) <= k):
            raise ValueError("RSS selection parameters out of range")
        _check_matrix(model.get("precedence"), k, k, "precedence")
        if any(not 0.0 < v < 1.0 for row in model["precedence"] for v in row):
            raise ValueError("RSS precedence entries must lie strictly between 0 and 1")

        self.model, self.checksum = model, checksum
        self.version = str(model.get("version", "unknown"))
        self.promotion = model.get("promotion")
        self.vocab = {g: i for i, g in enumerate(vocab_list)}
        self.keywords, self.hidden = keywords, hidden
        self.selection = dict(sel)
        self._w1, self._b1 = net["W1"], net["b1"]
        self._wp, self._bp, self._cal = prob["W"], prob["b"], prob["calibration"]
        self._wt, self._bt = (lat["W"], lat["b"]) if lat is not None else (None, None)
        self._precedence = model["precedence"]

    # -- inference ------------------------------------------------------------
    def _hidden_activations(self, features: Mapping[int, float]) -> list[float]:
        h = list(self._b1)
        for i, v in features.items():
            row = self._w1[i]
            for j in range(self.hidden):
                h[j] += v * row[j]
        return [x if x > 0.0 else 0.0 for x in h]

    def score(self, problem: Any, context: Optional[Mapping[str, Any]] = None) -> dict[str, dict[str, Any]]:
        """Per-style calibrated probability and (if trained) expected latency in ms."""
        h = self._hidden_activations(featurize(problem, context, vocab=self.vocab, keywords=self.keywords))
        out: dict[str, dict[str, Any]] = {}
        for k, name in enumerate(STRATEGIES):
            z = self._bp[k] + sum(w * a for w, a in zip(self._wp[k], h))
            a, b = self._cal[k]
            entry: dict[str, Any] = {"probability": _sigmoid(a * z + b), "expected_log_ms": None, "expected_ms": None}
            if self._wt is not None and self._bt is not None:
                weights, biases = self._wt, self._bt
                log_ms = biases[k] + sum(w * a2 for w, a2 in zip(weights[k], h))
                entry["expected_log_ms"] = log_ms
                entry["expected_ms"] = max(0.0, math.expm1(min(15.0, max(0.0, log_ms))))
            out[name] = entry
        return out

    def utilities(self, scores: Mapping[str, Mapping[str, Any]], names: Optional[Iterable[str]] = None) -> dict[str, float]:
        mu, cap = self.selection["latency_weight"], self.selection["latency_cap_ms"]
        return {n: strategy_utility(scores[n]["probability"], scores[n]["expected_log_ms"], mu, cap)
                for n in (names or STRATEGIES)}

    def select(self, problem: Any, context: Optional[Mapping[str, Any]] = None, *,
               allowed: Optional[Iterable[str]] = None, max_strategies: Optional[int] = None,
               fallback: Optional[Callable[[str], Mapping[str, Any]]] = None) -> dict[str, Any]:
        """Choose and order reasoning styles.

        ``allowed`` restricts candidates to styles the caller can actually dispatch (for
        example the keys of ``ReasoningTypes._TASK_TYPES``); ``max_strategies`` should be
        ``ReasoningTypes.max_combined_types``. When the best style's calibrated probability
        is below the checkpoint's ``min_confidence`` and ``fallback`` is given (typically
        ``ReasoningTypes.select_reasoning_strategy``), the fallback's decision is returned
        with the learned scores attached, but only if the fallback has evidence of its own
        (non-empty ``matches``). A blind default is not better than a weak learned guess, so
        in that case the learned choice stands, flagged ``low_confidence``.
        """
        allowed_set = None if allowed is None else set(allowed)
        names = [n for n in STRATEGIES if allowed_set is None or n in allowed_set]
        if not names:
            raise ValueError("No allowed reasoning strategy is known to the selector")
        cap = int(max_strategies) if max_strategies is not None else self.selection["max_strategies"]
        scores = self.score(problem, context)
        utilities = self.utilities(scores, names)
        best = max(names, key=lambda n: (utilities[n], n))
        confidence = scores[best]["probability"]
        details = {
            "scores": {n: {"probability": round(scores[n]["probability"], 6),
                           "expected_ms": None if scores[n]["expected_ms"] is None else round(scores[n]["expected_ms"], 3),
                           "utility": round(utilities[n], 6)} for n in names},
            "confidence": round(confidence, 6),
            "model_version": self.version,
            "model_sha256": self.checksum,
        }
        if fallback is not None and confidence < self.selection["min_confidence"]:
            deferred = dict(fallback(problem_text(problem)))
            if deferred.get("matches"):
                deferred.update(details)
                deferred["fallback_method"] = deferred.get("method")
                deferred["method"] = "learned_selector_fallback"
                return deferred
        chosen = choose_strategies(utilities, cap, self.selection["threshold"])
        sequence = order_sequence(chosen, self._precedence)
        return {"strategy": "+".join(sequence), "method": "learned_selector", "sequence": sequence,
                "matches": {n: details["scores"][n]["probability"] for n in sequence},
                "low_confidence": confidence < self.selection["min_confidence"], **details}


__all__ = ["ReasoningStrategySelector"]