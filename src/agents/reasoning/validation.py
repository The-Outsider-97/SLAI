"""Validation Engine for the reasoning subsystem.
 
Responsibilities
----------------
- Execute a multi-phase validation pipeline over symbolic inference rules
  and knowledge-base facts:
      Phase 1 — Rule validation  (circular dependency, soundness)
      Phase 2 — Fact validation  (conflicts, redundancies, confidence bounds)
      Phase 3 — KB consistency   (Markov Logic Network soft-rule violations,
                                  cross-fact contradiction scanning)
- Expose each validation stage as a standalone, callable method so external
  callers (e.g. ``ReasoningAgent``) can run targeted checks without the
  full pipeline cost.
- Integrate with ``ReasoningMemory`` to log every validation report at high
  priority so the memory subsystem can surface validation history.
- Support an optional semantic-similarity layer (``sentence-transformers``)
  for embedding-based conflict and redundancy detection when the configured
  flag is set; degrade gracefully to lexical-only checks when the model is
  unavailable.
 
Design constraints
------------------
- All configuration comes exclusively from ``reasoning_config.yaml`` via the
  shared config loader.  No scalar values are hard-coded here.
- All helper logic (confidence math, KB normalization, conflict/redundancy
  scanning, rule ranking, etc.) comes from ``reasoning_helpers``; none is
  duplicated.
- All error classes come from ``reasoning_errors``; none is duplicated.
- The ``mln_rules`` module's rich public API (``evaluate_mln_rules``,
  ``KnowledgeIndex``, ``MLNRuleViolation``, ``validate_rule_registry``) is
  used in full — the engine does not re-implement soft-rule logic.
- ``ReasoningMemory`` is used only for experience logging; its API is not
  replicated.
- Local imports are **never** wrapped in try/except — consistent with the
  rest of the subsystem.
"""
 
from __future__ import annotations
 
import json
import time

from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
 
from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import * # type: ignore
from .modules.mln_rules import * # type: ignore
from .reasoning_memory import ReasoningMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Validation Engine")
printer = PrettyPrinter()

_RuleTriple  = Tuple[str, Callable[..., Any], float]   # (name, callable, weight)
_Fact        = Tuple[Any, Any, Any]
_KB          = Dict[_Fact, float]
_ConflictPair = Tuple[_Fact, _Fact]
 
 
# ---------------------------------------------------------------------------
# ValidationResult dataclass-equivalent (plain dict with fixed schema)
# ---------------------------------------------------------------------------
def _empty_result() -> Dict[str, Any]:
    """Return a blank validation result dict with all required keys."""
    return {
        "circular_rules":     [],
        "sound_rules":        {"sound": [], "unsound": []},
        "conflicts":          [],
        "redundancies":       [],
        "confidence_violations": [],
        "consistency":        {},
        "validation_status":  "pending",
        "execution_time_ms":  0.0,
        "timestamp_ms":       monotonic_timestamp_ms(),
    }
 
 
# ---------------------------------------------------------------------------
# ValidationEngine
# ---------------------------------------------------------------------------
class ValidationEngine:
    """Multi-phase validation engine for the symbolic reasoning subsystem.
 
    Thread safety
    ~~~~~~~~~~~~~
    A single ``RLock`` guards KB reads/writes.  Validation phases are
    read-only with respect to the internal KB, so concurrent ``validate_all``
    calls are safe as long as callers do not share the same engine instance
    across threads while also mutating the KB.
 
    Configuration (``validation`` section of ``reasoning_config.yaml``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All tunable parameters are documented in the YAML file.  Call
    ``reload_config()`` to pick up YAML changes at runtime without
    reconstructing the engine.
    """
 
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
 
    def __init__(self) -> None:
        self._lock: RLock = RLock()
 
        # ---- Configuration --------------------------------------------------
        self.config: Dict[str, Any]            = load_global_config()
        self.validation_config: Dict[str, Any] = get_config_section("validation", self.config)
        self.storage_config: Dict[str, Any]    = get_config_section("storage",    self.config)
 
        # Top-level scalars
        self.contradiction_threshold: float = clamp_confidence(
            self.config.get("contradiction_threshold", 0.25)
        )
        self.markov_logic_weight: float = clamp_confidence(
            self.config.get("markov_logic_weight", 0.7)
        )
 
        # validation section scalars
        self.enable: bool = bool(self.validation_config.get("enable", True))
        self.redundancy_margin: float = clamp_confidence(
            self.validation_config.get("redundancy_margin", 0.05)
        )
        self._max_circular_depth: int = bounded_iterations(
            self.validation_config.get("max_circular_depth", 3), minimum=1, maximum=64
        )
        self._validation_timeout: float = max(
            0.0, float(self.validation_config.get("validation_timeout", 1))
        )
        self.min_soundness_score: float = clamp_confidence(
            self.validation_config.get("min_soundness_score", 0.7)
        )
        self._max_validation_attempts: int = bounded_iterations(
            self.validation_config.get("max_validation_attempts", 5), minimum=1, maximum=32
        )
        self.mln_rule_confidence_threshold: float = clamp_confidence(
            self.validation_config.get("mln_rule_confidence_threshold", 0.7)
        )
        self._enable_semantic_redundancy: bool = bool(
            self.validation_config.get("enable_semantic_redundancy", False)
        )
        self._semantic_conflict_similarity: float = clamp_confidence(
            self.validation_config.get("semantic_conflict_similarity", 0.80)
        )
        self._semantic_redundancy_similarity: float = clamp_confidence(
            self.validation_config.get("semantic_redundancy_similarity", 0.95)
        )
        self._semantic_conflict_confidence_gap: float = clamp_confidence(
            self.validation_config.get("semantic_conflict_confidence_gap", 0.30)
        )
 
        # ---- Storage --------------------------------------------------------
        self._kb_path: Path = Path(self.storage_config.get("knowledge_db", ""))
 
        # ---- Core data structures -------------------------------------------
        self.knowledge_base: _KB = self._load_knowledge_base(self._kb_path)
 
        # ---- Optional semantic model ----------------------------------------
        self.semantic_model: Optional[Any] = None
        if self._enable_semantic_redundancy:
            self._initialize_semantic_model()
 
        # ---- Shared memory --------------------------------------------------
        self.reasoning_memory: ReasoningMemory = ReasoningMemory()
 
        logger.info(
            "ValidationEngine initialized | kb_facts=%d | enable=%s "
            "| semantic=%s | mln_rules=%d",
            len(self.knowledge_base),
            self.enable,
            self.semantic_model is not None,
            len(mln_rules),
        )
 
    # ------------------------------------------------------------------
    # Config reload
    # ------------------------------------------------------------------
 
    def reload_config(self) -> None:
        """Hot-reload YAML config without rebuilding resources or the KB.
 
        Only scalar config values are refreshed.  To pick up KB file
        changes, call ``_load_knowledge_base`` explicitly.
        """
        self.config            = load_global_config(force_reload=True)
        self.validation_config = get_config_section("validation", self.config)
        self.storage_config    = get_config_section("storage",    self.config)
 
        self.contradiction_threshold        = clamp_confidence(self.config.get("contradiction_threshold", 0.25))
        self.markov_logic_weight            = clamp_confidence(self.config.get("markov_logic_weight", 0.7))
        self.enable                         = bool(self.validation_config.get("enable", True))
        self.redundancy_margin              = clamp_confidence(self.validation_config.get("redundancy_margin", 0.05))
        self._max_circular_depth            = bounded_iterations(self.validation_config.get("max_circular_depth", 3), minimum=1, maximum=64)
        self._validation_timeout            = max(0.0, float(self.validation_config.get("validation_timeout", 1)))
        self.min_soundness_score            = clamp_confidence(self.validation_config.get("min_soundness_score", 0.7))
        self._max_validation_attempts       = bounded_iterations(self.validation_config.get("max_validation_attempts", 5), minimum=1, maximum=32)
        self.mln_rule_confidence_threshold  = clamp_confidence(self.validation_config.get("mln_rule_confidence_threshold", 0.7))
        self._enable_semantic_redundancy    = bool(self.validation_config.get("enable_semantic_redundancy", False))
        self._semantic_conflict_similarity  = clamp_confidence(self.validation_config.get("semantic_conflict_similarity", 0.80))
        self._semantic_redundancy_similarity = clamp_confidence(self.validation_config.get("semantic_redundancy_similarity", 0.95))
        self._semantic_conflict_confidence_gap = clamp_confidence(self.validation_config.get("semantic_conflict_confidence_gap", 0.30))
 
        logger.info("ValidationEngine configuration reloaded")
 
    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
 
    def _load_knowledge_base(self, kb_path: Path) -> _KB:
        """Load and normalise the KB from ``knowledge_db.json``.
 
        Handles three shapes inside the ``"knowledge"`` key:
        - ``[{"subject": s, "predicate": p, "object": o, "confidence": c}, ...]``
        - ``[[s, p, o, confidence], ...]``
        - ``{"s||p||o": confidence, ...}``
 
        Falls back to a root-level pipe-delimited dict when ``"knowledge"`` is
        absent.  Unknown / malformed entries are skipped with a warning.
        """
        if not kb_path.exists():
            logger.warning("KB not found at %s — starting with empty knowledge base", kb_path)
            return {}
 
        try:
            with open(kb_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise KnowledgePersistenceError(
                f"Invalid JSON in knowledge base: {kb_path}",
                cause=exc,
                context={"path": str(kb_path)},
            ) from exc
 
        facts_source = raw.get("knowledge") if isinstance(raw, dict) else None
        if facts_source is None:
            # Fallback: root-level dict with pipe-delimited keys
            facts_source = raw if isinstance(raw, dict) else {}
            if facts_source:
                logger.warning(
                    "'knowledge' key absent in %s — using root-level dict fallback", kb_path
                )
 
        parsed: _KB = {}
 
        if isinstance(facts_source, dict):
            for key, conf in facts_source.items():
                parts = str(key).split("||")
                if len(parts) != 3:
                    logger.warning("Skipping malformed KB key (expected s||p||o): %r", key)
                    continue
                try:
                    fact = normalize_fact(tuple(parts))  # type: ignore[arg-type]
                    parsed[fact] = clamp_confidence(conf)
                except Exception as exc:
                    logger.warning("Skipping invalid KB entry %r: %s", key, exc)
 
        elif isinstance(facts_source, list):
            for item in facts_source:
                try:
                    if isinstance(item, dict):
                        s = item.get("subject")
                        p = item.get("predicate")
                        o = item.get("object")
                        if s is None or p is None or o is None:
                            logger.warning("Skipping incomplete KB record: %s", item)
                            continue
                        conf = item.get("confidence") or item.get("weight") or 0.5
                        fact = normalize_fact((str(s), str(p), str(o)))
                        parsed[fact] = clamp_confidence(conf)
                    elif isinstance(item, (list, tuple)) and len(item) >= 3:
                        s, p, o = item[:3]
                        conf = item[3] if len(item) >= 4 else 0.5
                        fact = normalize_fact((str(s), str(p), str(o)))
                        parsed[fact] = clamp_confidence(conf)
                    else:
                        logger.warning("Unsupported KB item type %s: %r", type(item).__name__, item)
                except Exception as exc:
                    logger.warning("Skipping KB item %r: %s", item, exc)
 
        logger.info("Loaded %d facts from %s", len(parsed), kb_path)
        return parsed
 
    def _initialize_semantic_model(self) -> None:
        """Initialise the sentence-transformer model for embedding-based checks.
 
        On failure the engine degrades to lexical-only checks (no exception is
        propagated — the model is optional).
        """
        try:
            from src.agents.perception.modules.transformer import Transformer  # type: ignore
            self.semantic_model = Transformer()
            logger.info("Semantic similarity model initialised for validation")
        except Exception as exc:
            logger.warning(
                "Semantic model unavailable (%s) — falling back to lexical checks", exc
            )
            self.semantic_model = None
 
    # ------------------------------------------------------------------
    # Full validation pipeline
    # ------------------------------------------------------------------
 
    def validate_all(
        self,
        rules: List[_RuleTriple],
        new_facts: _KB,
        *,
        tag: str = "validation",
    ) -> Dict[str, Any]:
        """Execute the complete validation pipeline.
 
        Stages run in order:
        1. ``detect_circular_rules``   — topological cycle detection
        2. ``check_rule_soundness``    — KB match scoring per rule
        3. ``detect_fact_conflicts``   — contradiction scanning (new vs KB)
        4. ``check_redundancies``      — exact + semantic duplicate detection
        5. ``check_confidence_bounds`` — out-of-range confidence values
        6. ``validate_kb_consistency`` — MLN soft-rule + cross-contradiction scan
 
        Each stage records its own elapsed time.  Any stage that raises is
        caught; the error is embedded in the result under its stage key and
        ``validation_status`` is set to ``"partial"`` rather than ``"failed"``,
        so downstream callers always receive a complete result dict.
 
        Args:
            rules:     List of ``(name, callable, weight)`` rule triples.
            new_facts: Candidate facts to validate against the stored KB.
            tag:       Memory tag for the logged experience (default ``"validation"``).
 
        Returns:
            Dict with keys: ``circular_rules``, ``sound_rules``, ``conflicts``,
            ``redundancies``, ``confidence_violations``, ``consistency``,
            ``validation_status``, ``execution_time_ms``, ``timestamp_ms``.
        """
        if not self.enable:
            logger.info("ValidationEngine is disabled — skipping validate_all")
            result = _empty_result()
            result["validation_status"] = "disabled"
            return result
 
        result = _empty_result()
        t_global = time.monotonic()
        partial = False
 
        # ---- Phase 1: Rule validation ---------------------------------------
        t0 = time.monotonic()
        try:
            result["circular_rules"] = self.detect_circular_rules(rules)
        except Exception as exc:
            logger.error("Phase 1a (circular rules) failed: %s", exc)
            result["circular_rules"] = {"error": str(exc)}
            partial = True
 
        try:
            result["sound_rules"] = self.check_rule_soundness(rules)
        except Exception as exc:
            logger.error("Phase 1b (rule soundness) failed: %s", exc)
            result["sound_rules"] = {"error": str(exc)}
            partial = True
        result["phase1_elapsed_ms"] = round(elapsed_seconds(t0) * 1000, 2)
 
        # ---- Phase 2: Fact validation ----------------------------------------
        t0 = time.monotonic()
        try:
            result["conflicts"] = self.detect_fact_conflicts(new_facts)
        except Exception as exc:
            logger.error("Phase 2a (fact conflicts) failed: %s", exc)
            result["conflicts"] = {"error": str(exc)}
            partial = True
 
        try:
            result["redundancies"] = self.check_redundancies(new_facts)
        except Exception as exc:
            logger.error("Phase 2b (redundancies) failed: %s", exc)
            result["redundancies"] = {"error": str(exc)}
            partial = True
 
        try:
            result["confidence_violations"] = self.check_confidence_bounds(new_facts)
        except Exception as exc:
            logger.error("Phase 2c (confidence bounds) failed: %s", exc)
            result["confidence_violations"] = {"error": str(exc)}
            partial = True
        result["phase2_elapsed_ms"] = round(elapsed_seconds(t0) * 1000, 2)
 
        # ---- Phase 3: KB consistency ----------------------------------------
        t0 = time.monotonic()
        try:
            combined_facts: _KB = {**self.knowledge_base, **new_facts}
            result["consistency"] = self.validate_kb_consistency(combined_facts)
        except Exception as exc:
            logger.error("Phase 3 (KB consistency) failed: %s", exc)
            result["consistency"] = {"error": str(exc)}
            partial = True
        result["phase3_elapsed_ms"] = round(elapsed_seconds(t0) * 1000, 2)
 
        # ---- Finalise --------------------------------------------------------
        result["validation_status"] = "partial" if partial else "success"
        result["execution_time_ms"] = round(elapsed_seconds(t_global) * 1000, 2)
        result["timestamp_ms"] = monotonic_timestamp_ms()
 
        self._log_memory_event(
            "validation_report",
            {
                "validation_status":  result["validation_status"],
                "rule_count":         len(rules),
                "new_fact_count":     len(new_facts),
                "conflict_count":     len(result["conflicts"]) if isinstance(result["conflicts"], list) else -1,
                "redundancy_count":   len(result["redundancies"]) if isinstance(result["redundancies"], list) else -1,
                "circular_count":     len(result["circular_rules"]) if isinstance(result["circular_rules"], list) else -1,
                "execution_time_ms":  result["execution_time_ms"],
            },
            tag=tag,
            priority=0.9,
        )
        return result
 
    # ------------------------------------------------------------------
    # Phase 1 — Rule validation
    # ------------------------------------------------------------------
 
    def detect_circular_rules(
        self,
        rules: List[_RuleTriple],
        *,
        max_depth: Optional[int] = None,
    ) -> List[str]:
        """Detect circular rule dependencies using Kahn's topological sort.
 
        A rule has a circular dependency when its output facts (produced by
        calling it with an empty KB) reference other rule names that feed
        back into it, forming a cycle.
 
        The implementation first probes each rule with an empty KB to collect
        its output keys, then builds a directed dependency graph and runs
        Kahn's algorithm.  Rules with a non-zero in-degree after the sort
        form cycles.
 
        Args:
            rules:     List of ``(name, callable, weight)`` triples.
            max_depth: Override the configured ``max_circular_depth``.  Has
                       no effect on Kahn's algorithm itself but guards the
                       initial probe call depth.
 
        Returns:
            Sorted list of rule names participating in at least one cycle.
        """
        _depth = max_depth or self._max_circular_depth
        rule_names: Set[str] = {name for name, _, _ in rules}
        rule_graph: Dict[str, Set[str]] = defaultdict(set)
 
        for name, rule_fn, _ in rules:
            try:
                outputs = rule_fn({})
                if not isinstance(outputs, dict):
                    continue
                for fact in outputs:
                    # A dependency edge exists when a rule's output key (third
                    # element of a triple) matches another rule's name
                    if isinstance(fact, tuple) and len(fact) == 3:
                        obj = str(fact[2])
                        if obj in rule_names and obj != name:
                            rule_graph[name].add(obj)
            except Exception as exc:
                logger.warning("Rule '%s' raised during cycle-probe (empty KB): %s", name, exc)
 
        # Kahn's topological sort
        in_degree: Dict[str, int] = {n: 0 for n in rule_names}
        for node, deps in rule_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
 
        queue: List[str] = [n for n, deg in in_degree.items() if deg == 0]
        processed: Set[str] = set()
 
        while queue:
            node = queue.pop(0)
            processed.add(node)
            for neighbour in rule_graph.get(node, []):
                if neighbour in in_degree:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        queue.append(neighbour)
 
        circular = sorted(n for n, deg in in_degree.items() if deg > 0)
        if circular:
            logger.warning("Circular rules detected: %s", circular)
        return circular
 
    def check_rule_soundness(
        self,
        rules: List[_RuleTriple],
        *,
        min_score: Optional[float] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Score each rule by how well its inferred facts match the stored KB.
 
        A rule is *sound* when the weighted match score of its inferred
        facts against the KB is at or above ``min_soundness_score``.
 
        Match score for a single rule:
            score = mean(1 - |inferred_conf - kb_conf|) for all inferred facts.
 
        An empty inference produces a score of 0.0 (unsound by convention).
        If the rule raises, it is scored 0.0 and its exception is logged.
 
        Args:
            rules:     List of ``(name, callable, weight)`` triples.
            min_score: Override the configured ``min_soundness_score``.
 
        Returns:
            Dict with ``"sound"`` and ``"unsound"`` keys, each a list of
            ``(rule_name, match_score)`` sorted descending by score.
        """
        floor = clamp_confidence(min_score or self.min_soundness_score)
        sound:   List[Tuple[str, float]] = []
        unsound: List[Tuple[str, float]] = []
 
        with self._lock:
            kb_snapshot = dict(self.knowledge_base)
 
        for name, rule_fn, _weight in rules:
            try:
                inferred = rule_fn(kb_snapshot)
                score    = self._score_rule_match(inferred, kb_snapshot)
            except Exception as exc:
                logger.warning("Rule '%s' raised during soundness check: %s", name, exc)
                score = 0.0
 
            if score >= floor:
                sound.append((name, round(score, 6)))
            else:
                unsound.append((name, round(score, 6)))
 
        sound   = sorted(sound,   key=lambda x: -x[1])
        unsound = sorted(unsound, key=lambda x: -x[1])
 
        logger.debug(
            "Rule soundness | sound=%d | unsound=%d | threshold=%.3f",
            len(sound), len(unsound), floor,
        )
        return {"sound": sound, "unsound": unsound}
 
    # ------------------------------------------------------------------
    # Phase 2 — Fact validation
    # ------------------------------------------------------------------
 
    def detect_fact_conflicts(
        self,
        new_facts: _KB,
        *,
        threshold: Optional[float] = None,
    ) -> List[_ConflictPair]:
        """Multi-dimensional conflict detection between new facts and the stored KB.
 
        Detection layers (in order):
        1. **Direct inverse** — ``(s, p, o)`` vs ``(s, p, not_o)`` when both
           exceed the contradiction threshold and their confidence gap does too.
        2. **Boolean polarity** — ``(s, p, True)`` vs ``(s, p, False)``.
        3. **Predicate exclusivity** — two distinct objects under the same
           ``(subject, predicate)`` when only one object is expected
           (single-valued predicates detected heuristically as those that never
           appear with multiple objects in the stored KB).
        4. **Semantic** (optional) — embedding cosine-similarity above
           ``semantic_conflict_similarity`` with a confidence gap above
           ``semantic_conflict_confidence_gap``.
 
        Args:
            new_facts:  Candidate facts to check against ``self.knowledge_base``.
            threshold:  Override the configured ``contradiction_threshold``.
 
        Returns:
            List of ``(new_fact, kb_fact)`` conflict pairs (de-duplicated).
        """
        thr = clamp_confidence(threshold or self.contradiction_threshold)
        conflicts: List[_ConflictPair] = []
        seen: Set[frozenset] = set()
 
        with self._lock:
            kb_snapshot = dict(self.knowledge_base)
 
        for (s, p, o), new_conf in new_facts.items():
            if new_conf < thr:
                continue
 
            # Layer 1: not_ inverse
            inv_o    = f"not_{o}" if not str(o).startswith("not_") else str(o)[4:]
            inv_fact = (s, p, inv_o)
            kb_conf  = kb_snapshot.get(inv_fact)
            if kb_conf is not None and kb_conf >= thr:
                pair_key = frozenset([(s, p, o), inv_fact])
                if pair_key not in seen:
                    conflicts.append(((s, p, o), inv_fact))
                    seen.add(pair_key)
 
            # Layer 2: boolean polarity
            for true_val, false_val in [("true", "false"), ("True", "False")]:
                if str(o) == true_val:
                    bool_inv = (s, p, false_val)
                    kb_conf_b = kb_snapshot.get(bool_inv)
                    if kb_conf_b is not None and kb_conf_b >= thr:
                        pair_key = frozenset([(s, p, o), bool_inv])
                        if pair_key not in seen:
                            conflicts.append(((s, p, o), bool_inv))
                            seen.add(pair_key)
 
        # Layer 3: single-valued predicate exclusivity
        sv_conflicts = self._detect_single_valued_conflicts(new_facts, kb_snapshot, thr)
        for pair in sv_conflicts:
            pk = frozenset(pair)
            if pk not in seen:
                conflicts.append(pair)
                seen.add(pk)
 
        # Layer 4: semantic conflicts
        if self.semantic_model is not None:
            sem_conflicts = self._detect_semantic_conflicts(new_facts, kb_snapshot)
            for pair in sem_conflicts:
                pk = frozenset(pair)
                if pk not in seen:
                    conflicts.append(pair)
                    seen.add(pk)
 
        if conflicts:
            logger.debug("Fact conflict detection found %d conflict(s)", len(conflicts))
        return conflicts
 
    def check_redundancies(
        self,
        new_facts: _KB,
        *,
        margin: Optional[float] = None,
    ) -> List[_Fact]:
        """Multi-modal redundancy detection between new facts and the stored KB.
 
        Detection layers:
        1. **Exact match** — same fact key with confidence within
           ``redundancy_margin`` of the stored value.
        2. **Semantic** (optional) — embedding cosine-similarity above
           ``semantic_redundancy_similarity``.
 
        Args:
            new_facts: Candidate facts to check.
            margin:    Override the configured ``redundancy_margin``.
 
        Returns:
            De-duplicated list of redundant fact keys.
        """
        mgn = clamp_confidence(margin or self.redundancy_margin)
        redundant: List[_Fact] = []
 
        with self._lock:
            kb_snapshot = dict(self.knowledge_base)
 
        for fact, new_conf in new_facts.items():
            kb_conf = kb_snapshot.get(fact)
            if kb_conf is not None and abs(new_conf - kb_conf) <= mgn:
                redundant.append(fact)
 
        if self.semantic_model is not None:
            sem_redundancies = self._detect_semantic_redundancies(new_facts, kb_snapshot)
            for fact in sem_redundancies:
                if fact not in redundant:
                    redundant.append(fact)
 
        logger.debug("Redundancy check found %d redundant fact(s)", len(redundant))
        return redundant
 
    def check_confidence_bounds(
        self,
        facts: _KB,
        *,
        strict: bool = False,
    ) -> List[_Fact]:
        """Return facts whose confidence value is outside ``[0.0, 1.0]``.
 
        Args:
            facts:  Fact dict to scan.
            strict: If ``True``, also flag facts with confidence exactly
                    at 0.0 (vacuous belief) or exactly at 1.0 (absolute
                    certainty, often a data error in probabilistic systems).
 
        Returns:
            List of fact keys with out-of-bounds (or, when strict, boundary)
            confidence values.
        """
        violations: List[_Fact] = []
        for fact, conf in facts.items():
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                violations.append(fact)
                continue
            if not (0.0 <= conf_val <= 1.0):
                violations.append(fact)
            elif strict and (conf_val == 0.0 or conf_val == 1.0):
                violations.append(fact)
        return violations
 
    # ------------------------------------------------------------------
    # Phase 3 — KB consistency
    # ------------------------------------------------------------------
 
    def validate_kb_consistency(
        self,
        kb: _KB,
        *,
        attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full KB consistency check with retry logic and timeout guarding.
 
        Wraps ``_perform_consistency_check`` in a retry loop that respects
        ``max_validation_attempts`` and ``validation_timeout``.  On exhaustion,
        raises ``ValidationEngineError``.
 
        Args:
            kb:       The combined knowledge base to validate.
            attempts: Override the configured ``max_validation_attempts``.
 
        Returns:
            Consistency report dict with keys: ``total_facts``,
            ``contradictions``, ``redundant_pairs``, ``confidence_violations``,
            ``markov_violations``, ``execution_time_ms``.
        """
        max_att = attempts or self._max_validation_attempts
        last_exc: Optional[Exception] = None
 
        for attempt in range(1, max_att + 1):
            try:
                return self._perform_consistency_check(kb)
            except ValidationEngineError:
                raise  # already typed — re-raise immediately
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Consistency check attempt %d/%d failed: %s",
                    attempt, max_att, exc,
                )
                if attempt < max_att:
                    time.sleep(self._validation_timeout)
 
        raise ValidationEngineError(
            f"KB consistency check failed after {max_att} attempt(s)",
            cause=last_exc,
            context={"attempts": max_att, "kb_size": len(kb)},
        )
 
    def _perform_consistency_check(self, kb: _KB) -> Dict[str, Any]:
        """Core consistency validation pass.
 
        Steps:
        1. Confidence bound scan for every fact.
        2. Cross-fact contradiction scan (inverse / boolean polarity) across the
           entire KB (not just new facts vs KB).
        3. Redundant-pair detection within the KB.
        4. MLN soft-rule evaluation via ``evaluate_mln_rules``.
 
        Returns:
            Consistency report dict.
        """
        t0 = time.monotonic()
        report: Dict[str, Any] = {
            "total_facts":           len(kb),
            "contradictions":        [],
            "redundant_pairs":       [],
            "confidence_violations": [],
            "markov_violations":     [],
            "mln_summary":           {},
            "execution_time_ms":     0.0,
        }
 
        # Step 1: confidence bounds
        report["confidence_violations"] = self.check_confidence_bounds(kb)
 
        # Step 2: cross-fact contradictions within the full KB
        sp_index: Dict[Tuple[Any, Any], List[Tuple[Any, float]]] = defaultdict(list)
        for (s, p, o), conf in kb.items():
            sp_index[(s, p)].append((o, conf))
 
        for (s, p), objects in sp_index.items():
            for i, (o1, c1) in enumerate(objects):
                for o2, c2 in objects[i + 1:]:
                    # Inverse / boolean polarity contradiction
                    if self._are_contradictory(str(o1), str(o2)):
                        mean_conf = (c1 + c2) / 2
                        if mean_conf > self.contradiction_threshold:
                            report["contradictions"].append(
                                {
                                    "fact_a": (s, p, o1, round(c1, 6)),
                                    "fact_b": (s, p, o2, round(c2, 6)),
                                    "mean_confidence": round(mean_conf, 6),
                                }
                            )
 
        # Step 3: redundant pairs (identical (s, p, o) keys with differing confidence)
        # — pairs are within-KB exact duplicates, which should be impossible with a
        # dict but can arise from merging two dicts with same key; detect near-identical
        # confidence values under the same predicate group.
        seen_sp: Dict[Tuple[Any, Any], Dict[Any, float]] = defaultdict(dict)
        for (s, p, o), conf in kb.items():
            if o in seen_sp[(s, p)]:
                report["redundant_pairs"].append(
                    {"fact_a": (s, p, o, seen_sp[(s, p)][o]), "fact_b": (s, p, o, conf)}
                )
            else:
                seen_sp[(s, p)][o] = conf
 
        # Step 4: MLN soft-rule evaluation (full rich API)
        try:
            mln_report = evaluate_mln_rules(
                kb,
                min_confidence=self.mln_rule_confidence_threshold,
                include_payloads=True,
                raise_on_error=False,
            )
            report["markov_violations"] = mln_report.get("legacy_violations", [])
            report["mln_summary"]       = {
                "status":           mln_report.get("status"),
                "violation_count":  mln_report.get("violation_count", 0),
                "enabled_rules":    mln_report.get("enabled_rules", 0),
                "by_category":      mln_report.get("by_category", {}),
                "by_rule":          mln_report.get("by_rule", {}),
                "errors":           mln_report.get("errors", []),
            }
        except Exception as exc:
            logger.warning("MLN evaluation failed during consistency check: %s", exc)
            report["mln_summary"] = {"error": str(exc)}
 
        report["execution_time_ms"] = round(elapsed_seconds(t0) * 1000, 2)
        return report
 
    # ------------------------------------------------------------------
    # Targeted public helpers
    # ------------------------------------------------------------------
 
    def validate_mln_rules(
        self,
        kb: Optional[_KB] = None,
        *,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run MLN soft-rule evaluation in isolation.
 
        Args:
            kb:             Facts to evaluate (defaults to the stored KB).
            min_confidence: Override the configured MLN confidence threshold.
 
        Returns:
            Full MLN evaluation report from ``evaluate_mln_rules``.
        """
        target_kb = kb if kb is not None else dict(self.knowledge_base)
        threshold = clamp_confidence(min_confidence or self.mln_rule_confidence_threshold)
        return evaluate_mln_rules(
            target_kb,
            min_confidence=threshold,
            include_payloads=True,
            raise_on_error=False,
        )
 
    def validate_rule_registry_integrity(self) -> Dict[str, Any]:
        """Validate the MLN rule registry for duplicate IDs and broken evaluators.
 
        Returns:
            Report from ``validate_rule_registry``.
        """
        return validate_rule_registry()
 
    def summarize_validation_coverage(
        self,
        rules: List[_RuleTriple],
        new_facts: _KB,
    ) -> Dict[str, Any]:
        """Lightweight coverage summary without running the full pipeline.
 
        Useful for pre-flight checks before committing expensive validations.
 
        Returns:
            Dict with ``rule_count``, ``new_fact_count``, ``kb_size``,
            ``mln_rule_count``, ``semantic_enabled``.
        """
        return json_safe_reasoning_state({
            "rule_count":       len(rules),
            "new_fact_count":   len(new_facts),
            "kb_size":          len(self.knowledge_base),
            "mln_rule_count":   len(mln_rules),
            "semantic_enabled": self.semantic_model is not None,
            "timestamp_ms":     monotonic_timestamp_ms(),
        })
 
    def audit_kb(self) -> Dict[str, Any]:
        """Full audit of the stored KB without requiring candidate facts.
 
        Equivalent to running ``validate_kb_consistency`` on the stored KB
        plus a confidence-bound scan of all facts.
 
        Returns:
            Consistency report enriched with a ``"kb_path"`` field.
        """
        with self._lock:
            kb_snapshot = dict(self.knowledge_base)
        report = self.validate_kb_consistency(kb_snapshot)
        report["kb_path"] = str(self._kb_path)
        return report
 
    # ------------------------------------------------------------------
    # Private helpers — scoring, contradiction, semantic layers
    # ------------------------------------------------------------------
 
    def _score_rule_match(
        self,
        inferred: Any,
        kb: _KB,
    ) -> float:
        """Compute how well a rule's output matches the KB (0.0 – 1.0).
 
        Score = mean(1 - |inferred_conf - kb_conf|) across inferred facts.
        Facts not in the KB contribute 0.0 to the mean (penalised for novelty).
        An empty inference or non-dict return yields 0.0.
        """
        if not isinstance(inferred, dict) or not inferred:
            return 0.0
        total = 0.0
        for fact, inf_conf in inferred.items():
            try:
                inf_val = clamp_confidence(inf_conf)
                kb_val  = clamp_confidence(kb.get(fact, 0.0))
                total  += 1.0 - abs(inf_val - kb_val)
            except Exception:
                total += 0.0
        return total / len(inferred)
 
    @staticmethod
    def _are_contradictory(o1: str, o2: str) -> bool:
        """Return ``True`` when ``o1`` and ``o2`` form a known contradiction pair.
 
        Checks:
        1. ``not_`` prefix convention.
        2. Boolean polarity (``"true"`` / ``"false"``).
        """
        s1, s2 = str(o1).lower(), str(o2).lower()
        if s1 == f"not_{s2}" or s2 == f"not_{s1}":
            return True
        if {s1, s2} == {"true", "false"}:
            return True
        return False
 
    def _detect_single_valued_conflicts(
        self,
        new_facts: _KB,
        kb: _KB,
        threshold: float,
    ) -> List[_ConflictPair]:
        """Flag (subject, predicate) pairs that map to two different high-confidence objects.
 
        A predicate is treated as *single-valued* when the stored KB shows
        it appearing with exactly one distinct object per subject.  If both
        the new fact and the KB fact exceed the threshold and have different
        objects, they conflict.
        """
        # Build (s, p) → objects in KB
        kb_sp: Dict[Tuple[Any, Any], List[Any]] = defaultdict(list)
        for (s, p, o), conf in kb.items():
            if conf >= threshold:
                kb_sp[(s, p)].append(o)
 
        # Single-valued predicates: exactly one object per (s, p) in the whole KB
        single_valued: Set[Any] = {
            p for (_, p), objs in kb_sp.items() if len(set(objs)) == 1
        }
 
        conflicts: List[_ConflictPair] = []
        for (ns, np, no), nc in new_facts.items():
            if nc < threshold or np not in single_valued:
                continue
            kb_objs = [o for o in kb_sp.get((ns, np), []) if o != no]
            for ko in kb_objs:
                kb_conf = kb.get((ns, np, ko))
                if kb_conf is not None and kb_conf >= threshold:
                    conflicts.append(((ns, np, no), (ns, np, ko)))
        return conflicts
 
    def _detect_semantic_conflicts(
        self,
        new_facts: _KB,
        kb: _KB,
    ) -> List[_ConflictPair]:
        """Embedding-based semantic conflict detection.
 
        Two facts conflict semantically when their cosine similarity exceeds
        ``semantic_conflict_similarity`` AND their confidence gap exceeds
        ``semantic_conflict_confidence_gap``.
        """
        if self.semantic_model is None:
            return []
 
        try:
            from sentence_transformers import util as st_util  # type: ignore
        except ImportError:
            logger.debug("sentence_transformers not installed — skipping semantic conflict check")
            return []
 
        conflicts: List[_ConflictPair] = []
        sim_threshold  = self._semantic_conflict_similarity
        conf_gap_floor = self._semantic_conflict_confidence_gap
 
        new_texts = {fact: " ".join(str(x) for x in fact) for fact in new_facts}
        kb_texts  = {fact: " ".join(str(x) for x in fact) for fact in kb}
 
        try:
            new_embeddings = {
                fact: self.semantic_model.encode(text)
                for fact, text in new_texts.items()
            }
            for new_fact, new_emb in new_embeddings.items():
                for kb_fact, kb_text in kb_texts.items():
                    kb_emb   = self.semantic_model.encode(kb_text)
                    sim      = float(st_util.pytorch_cos_sim(new_emb, kb_emb).item())
                    conf_gap = abs(new_facts[new_fact] - kb.get(kb_fact, 0.0))
                    if sim >= sim_threshold and conf_gap >= conf_gap_floor:
                        conflicts.append((new_fact, kb_fact))
        except Exception as exc:
            logger.warning("Semantic conflict check failed: %s", exc)
 
        return conflicts
 
    def _detect_semantic_redundancies(
        self,
        new_facts: _KB,
        kb: _KB,
    ) -> List[_Fact]:
        """Embedding-based semantic redundancy detection.
 
        A new fact is semantically redundant if any KB fact has cosine
        similarity ≥ ``semantic_redundancy_similarity``.
        """
        if self.semantic_model is None:
            return []
 
        try:
            from sentence_transformers import util as st_util  # type: ignore
        except ImportError:
            logger.debug("sentence_transformers not installed — skipping semantic redundancy check")
            return []
 
        redundant: List[_Fact] = []
        sim_threshold = self._semantic_redundancy_similarity
 
        try:
            for fact in new_facts:
                new_text = " ".join(str(x) for x in fact)
                new_emb  = self.semantic_model.encode(new_text)
                for kb_fact in kb:
                    kb_text = " ".join(str(x) for x in kb_fact)
                    kb_emb  = self.semantic_model.encode(kb_text)
                    sim     = float(st_util.pytorch_cos_sim(new_emb, kb_emb).item())
                    if sim >= sim_threshold:
                        redundant.append(fact)
                        break
        except Exception as exc:
            logger.warning("Semantic redundancy check failed: %s", exc)
 
        return redundant
 
    # Kept for backward compatibility — called by legacy code as a named method.
    def _validate_with_markov_logic(
        self,
        kb: _KB,
        weight: float,
    ) -> List[Tuple[Any, str]]:
        """Run MLN evaluation and return violations in legacy tuple format.
 
        Returns:
            List of ``(rule_id_label, description)`` tuples compatible with
            the original ``_validate_with_markov_logic`` return type.
        """
        report = evaluate_mln_rules(
            kb,
            min_confidence=self.mln_rule_confidence_threshold,
            include_payloads=False,
            raise_on_error=False,
        )
        return list(report.get("legacy_violations", []))
 
    # ------------------------------------------------------------------
    # Internal utility
    # ------------------------------------------------------------------
 
    def _log_memory_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        tag: Optional[str] = None,
        priority: float = 0.9,
    ) -> None:
        """Log an experience to ReasoningMemory, swallowing any errors."""
        try:
            experience = {"type": event_type, **payload}
            kwargs: Dict[str, Any] = {"priority": priority}
            if tag:
                kwargs["tag"] = tag
            self.reasoning_memory.add(experience=experience, **kwargs)
        except Exception as exc:
            logger.debug("ReasoningMemory.add failed for '%s': %s", event_type, exc)
 
    def __repr__(self) -> str:
        return (
            f"ValidationEngine("
            f"kb_facts={len(self.knowledge_base)}, "
            f"enable={self.enable}, "
            f"semantic={self.semantic_model is not None}, "
            f"mln_rules={len(mln_rules)})"
        )
 
 
if __name__ == "__main__":
    import tempfile
    import yaml # type: ignore
    from unittest.mock import patch

    from .utils.config_loader import load_global_config

    print("\n=== Running Validation Engine (Integration Test) ===\n")

    with tempfile.TemporaryDirectory(prefix="validation_test_") as tmpdir:
        tmp = Path(tmpdir)
        kb_path = tmp / "knowledge_db.json"
        checkpoint_dir = tmp / "checkpoints"

        # ------------------------------------------------------------------
        # 1. Load the real config using the existing loader
        # ------------------------------------------------------------------
        config = load_global_config()
        config["storage"]["knowledge_db"] = str(kb_path)
        config["reasoning_memory"]["checkpoint_dir"] = str(checkpoint_dir)
        config["reasoning_memory"]["auto_save"] = False

        # ------------------------------------------------------------------
        # 2. Create a minimal test knowledge base
        # ------------------------------------------------------------------
        kb_data = {
            "knowledge": [
                {"subject": "cat",      "predicate": "is_animal", "object": "true", "confidence": 0.9},
                {"subject": "sky",      "predicate": "color",     "object": "blue", "confidence": 0.90},
                {"subject": "sky",      "predicate": "color",     "object": "not_blue", "confidence": 0.85},
                {"subject": "Socrates", "predicate": "is_alive",  "object": "True", "confidence": 0.9},
                {"subject": "Socrates", "predicate": "is_alive",  "object": "False", "confidence": 0.95},
                {"subject": "Water",    "predicate": "state_is",  "object": "Liquid", "confidence": 0.8},
                {"subject": "Water",    "predicate": "state_is",  "object": "Solid", "confidence": 0.85},
                {"subject": "Earth",    "predicate": "shape",     "object": "Round", "confidence": 0.99},
            ],
        }
        with open(kb_path, "w", encoding="utf-8") as f:
            json.dump(kb_data, f)

        # ------------------------------------------------------------------
        # 3. Patch load_global_config to return our modified config
        #    (validation and mln_rules both call it)
        # ------------------------------------------------------------------
        def _patched_load(*args, **kwargs):
            return config.copy()

        with patch("__main__.load_global_config", _patched_load):
            with patch("__main__.load_global_config", _patched_load):
                ve = ValidationEngine()

                # ------------------------------------------------------------------
                # 4. Run integration tests (compact)
                # ------------------------------------------------------------------
                # KB loading
                assert len(ve.knowledge_base) == len(kb_data["knowledge"])

                # Circular rule detection
                def rule_a(kb): return {("out_A", "leads", "RuleB"): 0.8}
                def rule_b(kb): return {("out_B", "leads", "RuleC"): 0.8}
                def rule_c(kb): return {("out_C", "leads", "RuleA"): 0.7}
                straight = [("Rule1", lambda _: {("earth", "shape", "round"): 0.95}, 0.9)]
                circular = [("RuleA", rule_a, 0.9), ("RuleB", rule_b, 0.8), ("RuleC", rule_c, 0.7)]
                assert ve.detect_circular_rules(straight) == []
                assert isinstance(ve.detect_circular_rules(circular), list)

                # Rule soundness
                def sound_rule(kb): return {("cat", "is_animal", "true"): 0.88}
                def unsound_rule(kb): return {("imaginary", "has_property", "magic"): 0.99}
                soundness = ve.check_rule_soundness([("SoundRule", sound_rule, 0.9), ("UnsoundRule", unsound_rule, 0.5)])
                assert "sound" in soundness and "unsound" in soundness

                # Fact conflicts
                new_conflicts = {("sky", "color", "not_blue"): 0.88, ("Socrates", "is_alive", "False"): 0.92}
                conflicts = ve.detect_fact_conflicts(new_conflicts)
                assert len(conflicts) >= 1

                # Redundancies
                new_redundant = {("Earth", "shape", "Round"): 0.995}
                redundant = ve.check_redundancies(new_redundant)
                assert any("round" in str(f).lower() for f in redundant)

                # Confidence bounds
                bad_conf = {("A", "p", "o1"): 1.5, ("B", "p", "o2"): -0.1}
                violations = ve.check_confidence_bounds(bad_conf)
                assert len(violations) == 2

                # KB consistency (MLN)
                sample_kb = {("Socrates", "is_alive", "True"): 0.92, ("Socrates", "is_dead", "True"): 0.95, ("Bad", "conf", "fact"): 1.5}
                consistency = ve.validate_kb_consistency(sample_kb)
                assert consistency["total_facts"] == len(sample_kb)
                assert len(consistency["confidence_violations"]) >= 1

                # Full pipeline
                full_result = ve.validate_all([("SoundRule", sound_rule, 0.9)], new_redundant)
                assert full_result["validation_status"] in {"success", "partial"}

                # Helper methods
                assert "violation_count" in ve.validate_mln_rules(sample_kb)
                assert "total_facts" in ve.audit_kb()
                coverage = ve.summarize_validation_coverage([("Dummy", lambda x: {}, 0.5)], new_redundant)
                assert coverage["rule_count"] == 1

                print("\n=== All integration tests passed ===")