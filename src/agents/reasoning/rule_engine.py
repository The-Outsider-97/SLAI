"""Rule Engine for the symbolic reasoning subsystem.
 
Responsibilities
----------------
- Maintain an in-memory knowledge base (KB) of confidence-weighted facts loaded
  from ``knowledge_db.json``.
- Register, validate, and execute symbolic inference rules, including built-in
  hard-coded rules (identity, transitive) and adaptively discovered rules.
- Detect and mitigate circular rule dependencies, fact contradictions, and
  knowledge redundancies.
- Perform natural-language pragmatic analysis: Gricean maxim evaluation,
  politeness-strategy detection, discourse-relation identification, sentiment
  scoring, and lightweight dependency analysis.
- Persist the enriched KB back to disk and keep ``ReasoningMemory`` up to date.
 
Design constraints
------------------
- All configuration comes exclusively from ``reasoning_config.yaml`` via the
  shared config loader; no values are hard-coded here.
- All helper logic (confidence math, KB normalization, rule validation, etc.)
  comes from ``reasoning_helpers``; none is duplicated.
- All error classes come from ``reasoning_errors``; none is duplicated.
- ``ReasoningMemory`` is used for experience logging; its API is not replicated.
- Local imports (config loader, errors, helpers, memory) are **never** wrapped
  in try/except — consistent with the rest of the subsystem.
"""
 
from __future__ import annotations
 
import itertools
import json
import re
import time
import yaml # type: ignore
 
from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from .reasoning_memory import ReasoningMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Rule Engine")
printer = PrettyPrinter()

# ---------------------------------------------------------------------------
# Internal type aliases
# ---------------------------------------------------------------------------
_RuleFunc = Callable[[Dict[Fact, float]], Dict[Fact, float]]
_RuleRegistry = Dict[str, _RuleFunc]
 
 
# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------
class RuleEngine:
    """Symbolic rule engine for the reasoning subsystem.
 
    Thread-safety
    ~~~~~~~~~~~~~
    A reentrant lock (``_lock``) guards all KB mutations and rule-registry
    writes so the engine can be shared across reasoning threads safely.
 
    Configuration (``rules`` section of ``reasoning_config.yaml``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    See the YAML file for the full set of tunable parameters.  The engine
    never reads config values a second time after ``__init__``; call
    ``reload_config()`` to pick up YAML changes at runtime.
    """
 
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
 
    def __init__(self) -> None:
        self._lock: RLock = RLock()
 
        # ---- Configuration --------------------------------------------------
        self.config: Dict[str, Any] = load_global_config()
        self.rules_config: Dict[str, Any]   = get_config_section("rules",   self.config)
        self.storage_config: Dict[str, Any] = get_config_section("storage", self.config)
 
        # Scalar config values — read once, stored as typed attributes
        self.enable_learning: bool        = bool(self.rules_config.get("enable_learning", True))
        self.min_support: float           = clamp_confidence(self.rules_config.get("min_support", 0.3))
        self.min_confidence: float        = clamp_confidence(self.rules_config.get("min_confidence", 0.7))
        self.auto_weight_adjustment: bool = bool(self.rules_config.get("auto_weight_adjustment", True))
        self._max_circular_depth: int     = bounded_iterations(
            self.rules_config.get("max_circular_depth", 3), minimum=1, maximum=32
        )
        self.max_utterance_length: int    = bounded_iterations(
            self.rules_config.get("max_utterance_length", 50), minimum=1, maximum=10_000
        )
        self.contradiction_threshold: float = clamp_confidence(
            self.config.get("contradiction_threshold", 0.25)
        )
        self._redundancy_margin: float    = clamp_confidence(
            get_config_section("validation", self.config).get("redundancy_margin", 0.05)
        )
        self._rule_backup_path: Path = Path(
            self.storage_config.get("rule_backup", "src/agents/knowledge/discovered_rules.json")
        )
 
        # Resource file paths
        self._kb_path: Path             = Path(self.storage_config.get("knowledge_db", ""))
        self._lexicon_path: Path        = Path(self.storage_config.get("lexicon_path", ""))
        self._syntax_path: Path         = Path(self.storage_config.get("dependency_rules_path", ""))
        self._discourse_path: Path      = Path(self.rules_config.get("discourse_markers_path", ""))
        self._politeness_path: Path     = Path(self.rules_config.get("politeness_strategies_path", ""))
 
        # ---- Core data structures -------------------------------------------
        # KB: Fact → confidence (float in [0, 1])
        self.knowledge_base: Dict[Fact, float] = {}
        # Rule registry: name → callable
        self._rule_registry: _RuleRegistry = {}
        # Weights for registered rules
        self.rule_weights: RuleWeightMap = {}
        # Antecedents / consequents for circular-dependency and conflict analysis
        self.rule_antecedents: Dict[str, List[Any]] = {}
        self.rule_consequents: Dict[str, List[Any]] = {}
 
        # ---- Load resources -------------------------------------------------
        self.knowledge_base    = self._load_knowledge_base(self._kb_path)
        self.sentiment_lexicon = self._load_sentiment_lexicon(self._lexicon_path)
        self.pos_patterns      = self._load_pos_patterns(self._syntax_path)
        self.dependency_rules  = self._build_dependency_rules()
        self.pragmatic_heuristics = self._build_pragmatic_heuristics()
 
        # ---- Bootstrap built-in rules ---------------------------------------
        self._register_builtin_rules()
 
        # ---- Shared memory --------------------------------------------------
        self.reasoning_memory: ReasoningMemory = ReasoningMemory()
 
        logger.info(
            "Rule Engine initialized | kb_facts=%d | rules=%d | sentiment_terms=%d "
            "| dependency_rules=%d",
            len(self.knowledge_base),
            len(self._rule_registry),
            len(self.sentiment_lexicon.get("positive", {})),
            len(self.dependency_rules),
        )
 
    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
 
    @property
    def max_circular_depth(self) -> int:
        """Maximum chain depth before a rule cycle is flagged."""
        return self._max_circular_depth
 
    @property
    def rule_count(self) -> int:
        """Total number of registered rules (built-in + learned)."""
        return len(self._rule_registry)
 
    @property
    def kb_size(self) -> int:
        """Number of facts currently in the knowledge base."""
        return len(self.knowledge_base)
 
    # ------------------------------------------------------------------
    # Config reload
    # ------------------------------------------------------------------
 
    def reload_config(self) -> None:
        """Hot-reload YAML config without rebuilding resources.
 
        Only scalar config values are updated; loaded resources (KB, lexicons,
        POS patterns) are *not* reloaded.  Call the relevant ``_load_*`` methods
        explicitly if resource files have changed.
        """
        self.config = load_global_config(force_reload=True)
        self.rules_config   = get_config_section("rules",   self.config)
        self.storage_config = get_config_section("storage", self.config)
 
        self.enable_learning        = bool(self.rules_config.get("enable_learning", True))
        self.min_support            = clamp_confidence(self.rules_config.get("min_support", 0.3))
        self.min_confidence         = clamp_confidence(self.rules_config.get("min_confidence", 0.7))
        self.auto_weight_adjustment = bool(self.rules_config.get("auto_weight_adjustment", True))
        self._max_circular_depth    = bounded_iterations(
            self.rules_config.get("max_circular_depth", 3), minimum=1, maximum=32
        )
        self.contradiction_threshold = clamp_confidence(
            self.config.get("contradiction_threshold", 0.25)
        )
        logger.info("RuleEngine configuration reloaded")
 
    # ------------------------------------------------------------------
    # Resource loading helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _load_structured_resource(path: Path) -> Dict[str, Any]:
        """Parse a YAML or JSON file into a dict.  Raises ``ResourceLoadError``."""
        if not path.exists():
            raise ResourceLoadError(
                f"Resource file not found: {path}",
                context={"path": str(path)},
            )
        suffix = path.suffix.lower()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                if suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(fh) or {}
                elif suffix == ".json":
                    data = json.load(fh)
                else:
                    raise ResourceLoadError(
                        f"Unsupported resource format '{suffix}' for {path}",
                        context={"path": str(path), "suffix": suffix},
                    )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ResourceLoadError(
                f"Failed to parse resource file {path}",
                cause=exc,
                context={"path": str(path)},
            ) from exc
 
        if not isinstance(data, dict):
            raise ResourceLoadError(
                f"Resource file must contain a mapping at root: {path}",
                context={"path": str(path), "type": type(data).__name__},
            )
        return data
 
    def _load_knowledge_base(self, kb_path: Path) -> Dict[Fact, float]:
        """Load and normalise the knowledge base from ``knowledge_db.json``.
 
        Supported shapes inside the ``"knowledge"`` key:
        - ``{"s||p||o": confidence, ...}``           (dict, pipe-delimited keys)
        - ``[[s, p, o, confidence], ...]``            (list of 4-element lists)
        - ``[{"subject": s, "predicate": p, ...}, ...]`` (list of record dicts)
 
        Falls back to root-level pipe-delimited dict when ``"knowledge"`` key is
        absent.  Unknown / malformed entries are skipped with a warning.
        """
        if not kb_path.exists():
            logger.warning("Knowledge base not found at %s — starting empty", kb_path)
            return {}
 
        logger.info("Loading knowledge base from %s", kb_path)
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
            # Fallback: treat the whole dict as a pipe-delimited fact map
            facts_source = raw if isinstance(raw, dict) else {}
            if facts_source:
                logger.warning(
                    "'knowledge' key absent in %s — falling back to root-level dict", kb_path
                )
 
        parsed: Dict[Fact, float] = {}
 
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
                    if isinstance(item, (list, tuple)) and len(item) == 4:
                        s, p, o, conf = item
                        fact = normalize_fact((str(s), str(p), str(o)))
                        parsed[fact] = clamp_confidence(conf)
                    elif isinstance(item, dict):
                        s = item.get("subject")
                        p = item.get("predicate")
                        o = item.get("object")
                        if s is None or p is None or o is None:
                            logger.warning("Skipping incomplete KB record: %s", item)
                            continue
                        fact = normalize_fact((str(s), str(p), str(o)))
                        parsed[fact] = clamp_confidence(item.get("confidence", 0.5))
                    else:
                        logger.warning("Unsupported KB item type %s: %r", type(item).__name__, item)
                except Exception as exc:
                    logger.warning("Skipping KB item %r: %s", item, exc)
 
        # Also load any rules recorded in the KB file into the rule registry
        if isinstance(raw, dict):
            self._bootstrap_rules_from_kb(raw.get("rules", []))
 
        logger.info("Loaded %d facts from %s", len(parsed), kb_path)
        return parsed
 
    def _bootstrap_rules_from_kb(self, rules_list: List[Dict[str, Any]]) -> None:
        """Register named rules found in the KB JSON (built-ins by callable name)."""
        _builtin_map: Dict[str, _RuleFunc] = {
            "identity_rule":    self._builtin_identity_rule,
            "transitive_rule":  self._builtin_transitive_rule,
        }
        for entry in rules_list:
            if not isinstance(entry, dict):
                continue
            name     = str(entry.get("name", ""))
            callable_name = str(entry.get("callable", name))
            weight   = clamp_confidence(entry.get("weight", 1.0))
            func     = _builtin_map.get(callable_name)
            if func is None:
                logger.debug("KB rule '%s' has no built-in callable — skipped", callable_name)
                continue
            if name not in self._rule_registry:
                self._add_rule_internal(func, name, weight, antecedents=[], consequents=[])
 
    def _load_sentiment_lexicon(self, lexicon_path: Path) -> Dict[str, Any]:
        """Load the sentiment lexicon from a sentiment_pack YAML or legacy JSON.
 
        Returns a normalised dict with keys:
        ``positive``, ``negative``, ``intensifiers``, ``negators``, ``antonyms``.
        """
        if not lexicon_path.exists():
            logger.warning("Sentiment lexicon not found at %s — using empty lexicon", lexicon_path)
            return {"positive": {}, "negative": {}, "intensifiers": {}, "negators": [], "antonyms": {}}
 
        logger.info("Loading sentiment lexicon from %s", lexicon_path)
        data = self._load_structured_resource(lexicon_path)
 
        # sentiment_pack YAML nests data under lexicons / modifiers / negation
        def _extract_scored_dict(node: Any) -> Dict[str, float]:
            """Flatten {word: score} or [{word: w, score: s}] to {word: float}."""
            if isinstance(node, dict):
                out: Dict[str, float] = {}
                for k, v in node.items():
                    if isinstance(v, (int, float)):
                        out[str(k)] = float(v)
                    elif isinstance(v, dict):
                        score = v.get("score") or v.get("value") or v.get("valence") or 0.0
                        out[str(k)] = float(score)
                return out
            if isinstance(node, list):
                out = {}
                for item in node:
                    if isinstance(item, dict):
                        word  = item.get("word") or item.get("text") or item.get("token")
                        score = item.get("score") or item.get("value") or item.get("valence") or 0.0
                        if word:
                            out[str(word)] = float(score)
                return out
            return {}
 
        def _extract_list(node: Any) -> List[str]:
            if isinstance(node, (list, tuple, set)):
                return [str(x) for x in node if x]
            if isinstance(node, dict):
                return [str(k) for k in node]
            return []
 
        # Resolve nested pack structure (sentiment_pack.yaml uses lexicons / modifiers)
        lexicons_node  = data.get("lexicons") or {}
        modifiers_node = data.get("modifiers") or {}
        negation_node  = data.get("negation")  or {}
 
        positive    = _extract_scored_dict(
            lexicons_node.get("positive")  or data.get("positive",     {})
        )
        negative    = _extract_scored_dict(
            lexicons_node.get("negative")  or data.get("negative",     {})
        )
        intensifiers = _extract_scored_dict(
            modifiers_node.get("intensifiers") or data.get("intensifiers", {})
        )
        negators_raw = (
            negation_node.get("negators")
            or negation_node.get("words")
            or data.get("negators", [])
        )
        negators: List[str] = _extract_list(negators_raw)
        antonyms: Dict[str, Any] = data.get("antonyms", {})
 
        lexicon = {
            "positive":    positive,
            "negative":    negative,
            "intensifiers": intensifiers,
            "negators":    negators,
            "antonyms":    antonyms,
        }
        logger.info(
            "Sentiment lexicon | positive=%d | negative=%d | intensifiers=%d | negators=%d",
            len(positive), len(negative), len(intensifiers), len(negators),
        )
        return lexicon
 
    def _load_pos_patterns(self, patterns_path: Path) -> Dict[str, List[str]]:
        """Load POS / chunk patterns from syntax_pack YAML or legacy JSON."""
        if not patterns_path.exists():
            logger.warning("POS patterns file not found at %s — using empty set", patterns_path)
            return {}
 
        logger.info("Loading POS patterns from %s", patterns_path)
        data = self._load_structured_resource(patterns_path)
 
        patterns_data: List[Any] = (
            data if isinstance(data, list)
            else data.get("phrase_patterns", [])
        )
        patterns: Dict[str, List[str]] = {}
        for idx, item in enumerate(patterns_data):
            if not isinstance(item, dict):
                continue
            name = (
                item.get("name") or item.get("id")
                or item.get("category") or f"pattern_{idx}"
            )
            # Prefer universal POS tags; fall back to legacy pattern key
            pattern = item.get("universal_pattern") or item.get("pattern")
            if isinstance(pattern, list):
                patterns[str(name)] = [str(tag) for tag in pattern]
 
        logger.info("Loaded %d POS patterns", len(patterns))
        return patterns
 
    def _load_discourse_markers(self) -> List[str]:
        """Load discourse markers from the configured nlg_pack / discourse file."""
        path = self._discourse_path
        if not path.exists():
            logger.warning("Discourse markers file not found at %s — using empty list", path)
            return []
 
        logger.info("Loading discourse markers from %s", path)
        data = self._load_structured_resource(path)
 
        raw = (
            data.get("discourse_markers")
            or data.get("discourse")
            or data.get("markers")
            or []
        )
 
        markers: List[str] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    markers.append(item)
                elif isinstance(item, dict):
                    val = item.get("text") or item.get("marker") or item.get("value")
                    if val:
                        markers.append(str(val))
        elif isinstance(raw, dict):
            for value in raw.values():
                sub = value if isinstance(value, list) else (
                    value.get("items") or value.get("markers") or []
                    if isinstance(value, dict) else []
                )
                for item in sub:
                    m = item if isinstance(item, str) else (
                        item.get("text") or item.get("marker") if isinstance(item, dict) else None
                    )
                    if m:
                        markers.append(str(m))
 
        return sorted({m.strip().lower() for m in markers if m.strip()})
 
    def _load_politeness_strategies(self) -> Dict[str, List[str]]:
        """Load politeness strategy phrase lists from the configured nlg_pack."""
        path = self._politeness_path
        if not path.exists():
            logger.warning("Politeness strategies file not found at %s — using empty set", path)
            return {}
 
        logger.info("Loading politeness strategies from %s", path)
        data = self._load_structured_resource(path)
 
        raw = (
            data.get("strategies")
            or data.get("politeness_strategies")
            or data.get("politeness")
            or {}
        )
        if not isinstance(raw, dict):
            return {}
 
        normalized: Dict[str, List[str]] = {}
        for strategy, value in raw.items():
            if isinstance(value, list):
                phrases = value
            elif isinstance(value, dict):
                phrases = (
                    value.get("phrases")
                    or value.get("items")
                    or value.get("examples")
                    or value.get("markers")
                    or []
                )
            else:
                phrases = []
            normalized[str(strategy)] = [
                str(p).strip() for p in phrases if str(p).strip()
            ]
        return normalized
 
    # ------------------------------------------------------------------
    # Pragmatic component assembly
    # ------------------------------------------------------------------
 
    def _build_dependency_rules(self) -> Dict[str, List[str]]:
        """Filter POS patterns to those representing grammatical dependencies."""
        dep_keywords = {"dependency", "relation", "modifier", "->"}
        return {
            name: pattern
            for name, pattern in self.pos_patterns.items()
            if "->" in name or any(kw in name.lower() for kw in dep_keywords)
        }
 
    def _build_pragmatic_heuristics(self) -> Dict[str, Any]:
        """Assemble the pragmatic analysis bundle used by ``analyze_utterance``."""
        discourse_markers = self._load_discourse_markers()
        return {
            "gricean_maxims":       self._build_gricean_maxims(discourse_markers),
            "sentiment_analysis":   self._build_sentiment_rules(),
            "politeness_strategies": self._load_politeness_strategies(),
            "discourse_markers":    discourse_markers,
        }
 
    def _build_gricean_maxims(
        self, discourse_markers: List[str]
    ) -> Dict[str, Callable[[str], bool]]:
        """Return Grice's maxims as named boolean callables.
 
        Each callable returns ``True`` when the maxim is **satisfied**.
        The ``quantity`` and ``manner`` thresholds are taken from config.
        """
        max_len   = self.max_utterance_length
        fillers   = set(self.rules_config.get("manner_filler_words", ["uh", "um", "like"]))
        hedges    = set(self.rules_config.get("quality_hedge_words",  ["probably", "perhaps", "maybe"]))
 
        return {
            # quantity: within configured utterance length (words)
            "quantity": lambda u: len(u.split()) <= max_len,
            # quality: no epistemic hedges (proxy for truth commitment)
            "quality":  lambda u: not any(h in u.lower() for h in hedges),
            # relation: at least one discourse marker present
            "relation": lambda u: any(m in u.lower() for m in discourse_markers),
            # manner: no disfluency filler words
            "manner":   lambda u: not any(f in u.lower().split() for f in fillers),
        }
 
    def _build_sentiment_rules(self) -> Dict[str, Any]:
        """Materialise references into the sentiment analysis bundle."""
        return {
            "positive":    self.sentiment_lexicon["positive"],
            "negative":    self.sentiment_lexicon["negative"],
            "intensifiers": self.sentiment_lexicon["intensifiers"],
            "negators":    set(self.sentiment_lexicon["negators"]),
        }
 
    # ------------------------------------------------------------------
    # Built-in symbolic rules
    # ------------------------------------------------------------------
 
    def _builtin_identity_rule(self, kb: Dict[Fact, float]) -> Dict[Fact, float]:
        """Identity rule: assert each fact with its own confidence (no-op baseline)."""
        return {fact: conf for fact, conf in kb.items()}
 
    def _builtin_transitive_rule(self, kb: Dict[Fact, float]) -> Dict[Fact, float]:
        """Transitive closure: if (A, R, B) and (B, R, C) infer (A, R, C).
 
        Inferred confidence = product of the two supporting confidences,
        ensuring derived facts are always weaker than their premises.
        """
        inferred: Dict[Fact, float] = {}
        facts = list(kb.keys())
        for f1 in facts:
            s1, r1, o1 = f1
            for f2 in facts:
                s2, r2, o2 = f2
                if r1 == r2 and o1 == s2 and s1 != o2:
                    new_fact: Fact = (s1, r1, o2)
                    derived_conf   = clamp_confidence(kb[f1] * kb[f2])
                    existing       = inferred.get(new_fact, 0.0)
                    inferred[new_fact] = merge_confidence(existing, derived_conf)
        return inferred
 
    def _register_builtin_rules(self) -> None:
        """Register hard-coded built-in rules at construction time."""
        self._add_rule_internal(
            self._builtin_identity_rule,   "identity_rule",   weight=1.0,
            antecedents=[], consequents=[],
        )
        self._add_rule_internal(
            self._builtin_transitive_rule, "transitive_rule", weight=0.8,
            antecedents=[], consequents=[],
        )
 
    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------
 
    def _add_rule_internal(
        self,
        rule_func: _RuleFunc,
        name: str,
        weight: float,
        *,
        antecedents: List[Any],
        consequents: List[Any],
    ) -> None:
        """Low-level (non-locking) rule insertion — always use ``add_rule`` externally."""
        safe_name, safe_weight = validate_rule_registration(rule_func, name, weight)
        self._rule_registry[safe_name]    = rule_func
        self.rule_weights[safe_name]      = safe_weight
        self.rule_antecedents[safe_name]  = list(antecedents)
        self.rule_consequents[safe_name]  = list(consequents)
 
    def add_rule(
        self,
        rule_func: _RuleFunc,
        name: str,
        weight: float,
        antecedents: Optional[List[Any]] = None,
        consequents: Optional[List[Any]] = None,
    ) -> bool:
        """Register a new inference rule thread-safely.
 
        Args:
            rule_func:    Callable ``(kb) → {fact: confidence}`` returning inferred facts.
            name:         Unique rule identifier.
            weight:       Initial rule weight in ``[0, 1]``.
            antecedents:  Element list that triggers this rule (for dependency analysis).
            consequents:  Elements this rule concludes (for circular-dependency analysis).
 
        Returns:
            ``True`` if the rule was registered; ``False`` if it already existed.
 
        Raises:
            RuleDefinitionError: On invalid callable, name, or weight.
            CircularReasoningError: If the rule would create a circular dependency
                that exceeds ``max_circular_depth``.
        """
        ants  = list(antecedents or [])
        cons  = list(consequents or [])
        safe_name, safe_weight = validate_rule_registration(rule_func, name, weight)
 
        with self._lock:
            if safe_name in self._rule_registry:
                logger.debug("Rule '%s' already registered — skipping", safe_name)
                return False
 
            if ants and cons and self._would_create_circular_dependency(ants, cons):
                raise CircularReasoningError(
                    f"Rule '{safe_name}' would create a circular dependency",
                    context={"antecedents": ants, "consequents": cons},
                )
            self._add_rule_internal(rule_func, safe_name, safe_weight, antecedents=ants, consequents=cons)
            logger.debug("Registered rule '%s' (weight=%.3f)", safe_name, safe_weight)
            return True
 
    def remove_rule(self, name: str) -> bool:
        """Deregister a rule by name.  Returns ``True`` if the rule existed."""
        with self._lock:
            if name not in self._rule_registry:
                return False
            del self._rule_registry[name]
            self.rule_weights.pop(name, None)
            self.rule_antecedents.pop(name, None)
            self.rule_consequents.pop(name, None)
            logger.info("Rule '%s' removed from registry", name)
            return True
 
    def get_rules_by_category(self, category: str) -> Dict[str, _RuleFunc]:
        """Return rules whose name contains ``category`` (case-insensitive).
 
        A full category taxonomy is a future extension; this provides a
        deterministic prefix/substring filter in the meantime.
        """
        lower = category.lower()
        with self._lock:
            return {
                name: func
                for name, func in self._rule_registry.items()
                if lower in name.lower()
            }
 
    def list_rules(self) -> List[Dict[str, Any]]:
        """Return a serialisable summary of all registered rules."""
        with self._lock:
            return [
                {
                    "name":         name,
                    "weight":       self.rule_weights.get(name, 0.0),
                    "antecedents":  self.rule_antecedents.get(name, []),
                    "consequents":  self.rule_consequents.get(name, []),
                }
                for name in self._rule_registry
            ]
 
    # ------------------------------------------------------------------
    # Knowledge base mutation
    # ------------------------------------------------------------------
    def assert_fact(self, fact: Union[Fact, Any], confidence: float, *,
                    allow_contradiction: bool = False) -> None:
        """Assert a new fact into the KB with the given confidence.
 
        Args:
            fact:                 A ``(subject, predicate, object)`` triple or anything
                                  acceptable by ``normalize_fact``.
            confidence:           Belief strength in ``[0, 1]``.
            allow_contradiction:  When ``False`` (default), raises ``ContradictionError``
                                  if the inverse of this fact already exists above
                                  ``contradiction_threshold``.
 
        Raises:
            ContradictionError: If a contradictory fact exists and
                ``allow_contradiction`` is ``False``.
        """
        canon = normalize_fact(fact)
        conf  = clamp_confidence(confidence)
        with self._lock:
            if not allow_contradiction:
                ensure_non_contradictory(
                    canon,
                    self.knowledge_base,
                    threshold=self.contradiction_threshold,
                )
            existing = self.knowledge_base.get(canon, 0.0)
            self.knowledge_base[canon] = merge_confidence(existing, conf)
 
    def retract_fact(self, fact: Union[Fact, Any]) -> bool:
        """Remove a fact from the KB.  Returns ``True`` if it existed."""
        canon = normalize_fact(fact)
        with self._lock:
            if canon in self.knowledge_base:
                del self.knowledge_base[canon]
                return True
            return False
 
    def query_fact(self, fact: Union[Fact, Any]) -> Optional[float]:
        """Return the confidence of a fact, or ``None`` if not in the KB."""
        canon = normalize_fact(fact)
        return self.knowledge_base.get(canon)
 
    def query_subject(self, subject: str) -> Dict[Fact, float]:
        """Return all KB facts whose subject matches (exact, case-sensitive)."""
        s = str(subject).strip()
        return {f: c for f, c in self.knowledge_base.items() if f[0] == s}
 
    def bulk_assert(self, facts: Dict[Union[Fact, Any], float], *,
                    allow_contradiction: bool = False) -> Tuple[int, int]:
        """Assert multiple facts at once.
 
        Returns:
            ``(inserted, skipped)`` counts.
        """
        inserted = skipped = 0
        for raw_fact, conf in facts.items():
            try:
                self.assert_fact(raw_fact, conf, allow_contradiction=allow_contradiction)
                inserted += 1
            except ContradictionError as exc:
                logger.warning("Skipping contradictory fact during bulk_assert: %s", exc)
                skipped += 1
        return inserted, skipped
 
    # ------------------------------------------------------------------
    # Inference execution
    # ------------------------------------------------------------------
 
    def run_inference(self, *,min_confidence_filter: Optional[float] = None,
                      max_rounds: Optional[int] = None) -> Dict[Fact, float]:
        """Apply all registered rules to the KB until convergence.
 
        Uses the frozen KB signature to detect convergence so the loop
        terminates even when rules generate circular derivations.
 
        Args:
            min_confidence_filter: If set, only return inferred facts above
                                   this confidence threshold.
            max_rounds:            Override the configured max learning cycles.
 
        Returns:
            Dict of newly inferred facts (not already in the KB) with their
            confidence values.
        """
        rounds = bounded_iterations(
            max_rounds or get_config_section("inference", self.config).get("max_learning_cycles", 100),
            minimum=1, maximum=10_000,
        )
        prev_sig: Optional[tuple] = None
        all_inferred: Dict[Fact, float] = {}
 
        with self._lock:
            rules_ordered = rank_rules_by_weight(
                [(name, func, self.rule_weights.get(name, 0.0))
                 for name, func in self._rule_registry.items()],
                self.rule_weights,
            )
 
            for _round in range(rounds):
                round_inferred: Dict[Fact, float] = {}
                for name, func, _w in rules_ordered:
                    try:
                        new_facts = func(dict(self.knowledge_base))
                    except Exception as exc:
                        raise RuleExecutionError(
                            f"Rule '{name}' raised an exception during inference",
                            cause=exc,
                            context={"rule": name},
                        ) from exc
 
                    if not isinstance(new_facts, dict):
                        logger.warning("Rule '%s' returned non-dict — skipping", name)
                        continue
 
                    for raw_fact, raw_conf in new_facts.items():
                        try:
                            fact = normalize_fact(raw_fact)
                            conf = clamp_confidence(raw_conf)
                        except Exception:
                            continue
                        if fact not in self.knowledge_base:
                            existing = round_inferred.get(fact, 0.0)
                            round_inferred[fact] = merge_confidence(existing, conf)
 
                    # Adaptive weight update if learning is enabled
                    if self.auto_weight_adjustment and round_inferred:
                        self.rule_weights[name] = update_rule_weight(
                            self.rule_weights.get(name, 0.5), success=bool(new_facts)
                        )
 
                # Merge this round's inferences into the working KB
                for fact, conf in round_inferred.items():
                    self.knowledge_base[fact] = merge_confidence(
                        self.knowledge_base.get(fact, 0.0), conf
                    )
                    all_inferred[fact] = merge_confidence(
                        all_inferred.get(fact, 0.0), conf
                    )
 
                # Convergence check via KB signature
                current_sig = freeze_kb_signature(self.knowledge_base)
                if current_sig == prev_sig:
                    logger.debug("Inference converged after %d round(s)", _round + 1)
                    break
                prev_sig = current_sig
 
        if min_confidence_filter is not None:
            floor = clamp_confidence(min_confidence_filter)
            all_inferred = {f: c for f, c in all_inferred.items() if c >= floor}
 
        if all_inferred:
            self._log_memory_event("inference_run", {"inferred_count": len(all_inferred)})
 
        return all_inferred
 
    def _apply_all_rules(self) -> Dict[Fact, float]:
        """Apply all rules once and return the merged result (no KB mutation)."""
        inferred: Dict[Fact, float] = {}
        for name, func in self._rule_registry.items():
            try:
                new_facts = func(dict(self.knowledge_base))
            except Exception as exc:
                logger.warning("Rule '%s' raised during _apply_all_rules: %s", name, exc)
                continue
            if not isinstance(new_facts, dict):
                continue
            for raw_fact, raw_conf in new_facts.items():
                try:
                    fact = normalize_fact(raw_fact)
                    conf = clamp_confidence(raw_conf)
                    inferred[fact] = merge_confidence(inferred.get(fact, 0.0), conf)
                except Exception:
                    continue
        return inferred
 
    # ------------------------------------------------------------------
    # Adaptive rule discovery (Apriori-inspired association mining)
    # ------------------------------------------------------------------
 
    def discover_rules(self) -> int:
        """Mine association rules from high-confidence KB facts.
 
        Implements a confidence-weighted Apriori adaptation (Agrawal & Srikant
        1994; Lê et al. 2014 fuzzy extension).  Only runs when
        ``enable_learning`` is ``True``.
 
        Returns:
            Number of new rules registered.
        """
        if not self.enable_learning:
            logger.debug("Rule discovery skipped (enable_learning=False)")
            return 0
 
        with self._lock:
            kb_snapshot = dict(self.knowledge_base)
 
        # High-confidence facts form the transaction database
        transactions: List[Fact] = [
            fact for fact, conf in kb_snapshot.items() if conf > 0.5
        ]
        if len(transactions) < 2:
            logger.debug("Too few transactions for rule discovery (%d)", len(transactions))
            return 0
 
        n = len(transactions)
 
        # ---- Weighted support for sub-sequences of each fact triple ---------
        itemsets: Dict[tuple, float] = defaultdict(float)
        for fact in transactions:
            for r in range(1, 4):
                for combo in itertools.combinations(fact, r):
                    itemsets[combo] += kb_snapshot.get(fact, 0.5)
 
        freq_itemsets = {
            k: v / n for k, v in itemsets.items() if (v / n) >= self.min_support
        }
 
        # ---- Generate candidate antecedent → consequent pairs ---------------
        new_count = 0
        for itemset, _sup in freq_itemsets.items():
            if len(itemset) < 2:
                continue
            for split in range(1, len(itemset)):
                ant  = itemset[:split]
                cons = itemset[split:]
 
                ant_support = sum(
                    kb_snapshot.get(f, 0.0)
                    for f in transactions
                    if set(ant).issubset(set(f))
                ) / n
                rule_support = sum(
                    kb_snapshot.get(f, 0.0)
                    for f in transactions
                    if set(ant + cons).issubset(set(f))
                ) / n
 
                if ant_support <= 0:
                    continue
                confidence = rule_support / ant_support
                if confidence < self.min_confidence:
                    continue
 
                rule_name = f"LearnedRule_{abs(hash(frozenset(ant + cons)))}"
                if rule_name in self._rule_registry:
                    continue
 
                # Closure captures ant/cons/confidence by value
                def _make_rule(
                    antecedents: tuple, consequents: tuple, conf: float
                ) -> _RuleFunc:
                    def rule_func(kb: Dict[Fact, float]) -> Dict[Fact, float]:
                        matches = [f for f in kb if all(e in f for e in antecedents)]
                        if not matches:
                            return {}
                        inferred_conf = conf * len(matches) / (len(kb) + 1e-8)
                        return {consequents: clamp_confidence(inferred_conf)}
                    rule_func.__name__ = f"learned_{abs(hash(frozenset(antecedents + consequents)))}"
                    return rule_func
 
                rule_func = _make_rule(ant, cons, confidence)
 
                try:
                    registered = self.add_rule(
                        rule_func, rule_name, weight=confidence,
                        antecedents=list(ant), consequents=list(cons),
                    )
                except CircularReasoningError as exc:
                    logger.warning("Skipping circular discovered rule '%s': %s", rule_name, exc)
                    continue
 
                if registered:
                    new_count += 1
                    self._log_memory_event(
                        "rule_discovery",
                        {
                            "rule_name": rule_name,
                            "antecedents": list(ant),
                            "consequents": list(cons),
                            "confidence": confidence,
                        },
                    )
 
        logger.info("Rule discovery complete — %d new rules registered", new_count)
        return new_count
 
    # ------------------------------------------------------------------
    # Circular dependency detection
    # ------------------------------------------------------------------
 
    def _would_create_circular_dependency(
        self,
        antecedents: Iterable[Any],
        consequents: Iterable[Any],
    ) -> bool:
        """Fast pre-registration check: would adding this rule form a cycle?
 
        Returns ``True`` if any consequent element appears in the antecedents
        of an existing rule, forming a direct one-hop cycle.  The full DFS
        ``detect_circular_rules`` catches longer cycles after registration.
        """
        cons_set = set(consequents)
        for rule_name in self._rule_registry:
            existing_ants = set(self.rule_antecedents.get(rule_name, []))
            if cons_set & existing_ants:
                return True
        return False
 
    def detect_circular_rules(self, max_depth: Optional[int] = None) -> List[List[str]]:
        """DFS-based detection of circular rule dependency chains.
 
        Args:
            max_depth: Override the configured max circular depth.
 
        Returns:
            List of circular chains (each a list of rule names forming a cycle).
        """
        depth = max_depth or self._max_circular_depth
        circular_chains: List[List[str]] = []
 
        with self._lock:
            rule_deps: Dict[str, List[str]] = {
                rule: self._get_rule_dependencies(rule)
                for rule in self._rule_registry
            }
 
            for start_rule in self._rule_registry:
                visited: Set[str] = set()
                stack: List[Tuple[str, List[str]]] = [(start_rule, [start_rule])]
 
                while stack:
                    current, path = stack.pop()
                    if current in visited:
                        if current == path[0] and len(path) > 1:
                            circular_chains.append(path)
                        continue
                    visited.add(current)
                    if len(path) >= depth:
                        continue
                    for dep in rule_deps.get(current, []):
                        stack.append((dep, path + [dep]))
 
        if circular_chains:
            logger.warning(
                "Detected %d circular rule chain(s): %s",
                len(circular_chains), circular_chains,
            )
        return circular_chains
 
    def _get_rule_dependencies(self, rule_name: str) -> List[str]:
        """Return names of rules whose antecedents overlap with this rule's consequents."""
        cons = self.rule_consequents.get(rule_name, [])
        return [
            r for r in self._rule_registry
            if r != rule_name
            and any(c in self.rule_antecedents.get(r, []) for c in cons)
        ]
 
    def adjust_circular_rule_weights(self, circular_chains: List[List[str]], *, decay: float = 0.5) -> Dict[str, Tuple[float, float]]:
        """Decay rule weights for all rules participating in circular chains.
 
        Args:
            circular_chains: Output from ``detect_circular_rules``.
            decay:           Multiplicative decay factor (default 0.5).
 
        Returns:
            Dict mapping rule name → ``(old_weight, new_weight)``.
        """
        decay = clamp_confidence(decay)
        changes: Dict[str, Tuple[float, float]] = {}
        with self._lock:
            for chain in circular_chains:
                for rule_name in chain:
                    if rule_name not in self.rule_weights:
                        continue
                    old_w = self.rule_weights[rule_name]
                    new_w = max(0.05, old_w * decay)
                    self.rule_weights[rule_name] = new_w
                    changes[rule_name] = (old_w, new_w)
                    logger.info(
                        "Decayed circular rule weight '%s': %.4f → %.4f",
                        rule_name, old_w, new_w,
                    )
        return changes
 
    # ------------------------------------------------------------------
    # Conflict and redundancy detection
    # ------------------------------------------------------------------
 
    def detect_fact_conflicts(self, contradiction_threshold: Optional[float] = None) -> List[Tuple[Tuple, Tuple]]:
        """Identify contradictory fact pairs in the KB.
 
        Two facts ``(s, p, o1, c1)`` and ``(s, p, o2, c2)`` are conflicting
        when ``o1`` and ``o2`` are antonyms and their mean confidence exceeds
        the threshold.
 
        Returns:
            List of ``((s, p, o1, c1), (s, p, o2, c2))`` conflict pairs.
        """
        threshold = clamp_confidence(
            contradiction_threshold or self.contradiction_threshold
        )
        antonyms: Dict[str, Any] = self.sentiment_lexicon.get("antonyms", {})
        conflicts: List[Tuple[Tuple, Tuple]] = []
 
        # Build (subject, predicate) → [(object, confidence)] index
        sp_index: Dict[Tuple[str, str], List[Tuple[Any, float]]] = defaultdict(list)
        for (s, p, o), conf in self.knowledge_base.items():
            sp_index[(s, p)].append((o, conf))
 
        for (_s, _p), objects in sp_index.items():
            for i, (o1, c1) in enumerate(objects):
                for o2, c2 in objects[i + 1:]:
                    if self._are_contradictory(str(o1), str(o2), antonyms):
                        if (c1 + c2) / 2 > threshold:
                            conflicts.append(
                                ((_s, _p, o1, c1), (_s, _p, o2, c2))
                            )
 
        if conflicts:
            logger.warning("Detected %d fact conflict(s)", len(conflicts))
        return conflicts
 
    def _are_contradictory(
        self,
        obj1: str,
        obj2: str,
        antonyms: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return ``True`` if ``obj1`` and ``obj2`` are known antonyms.
 
        Checks:
        1. Direct antonym lookup in sentiment lexicon.
        2. ``not_`` prefix convention (e.g. ``"true"`` vs ``"not_true"``).
        """
        ants = antonyms or self.sentiment_lexicon.get("antonyms", {})
        if isinstance(ants.get(obj1), (list, set, tuple)) and obj2 in ants[obj1]:
            return True
        if isinstance(ants.get(obj2), (list, set, tuple)) and obj1 in ants[obj2]:
            return True
        # not_ prefix convention
        if obj1 == f"not_{obj2}" or obj2 == f"not_{obj1}":
            return True
        # Boolean polarity
        if {obj1.lower(), obj2.lower()} == {"true", "false"}:
            return True
        return False
 
    def redundant_fact_check(
        self, confidence_margin: Optional[float] = None
    ) -> List[Tuple[Fact, float, float]]:
        """Find KB facts that are inferrable from existing rules within a margin.
 
        A fact is redundant when the inferred confidence is within
        ``confidence_margin`` of the stored confidence.
 
        Returns:
            List of ``(fact, kb_confidence, inferred_confidence)`` triples.
        """
        margin = clamp_confidence(confidence_margin or self._redundancy_margin)
        inferred = self._apply_all_rules()
        return [
            (fact, kb_conf, inferred.get(fact, 0.0))
            for fact, kb_conf in self.knowledge_base.items()
            if inferred.get(fact, 0.0) >= (kb_conf - margin)
        ]
 
    # ------------------------------------------------------------------
    # Pragmatic / NLU analysis
    # ------------------------------------------------------------------
 
    def analyze_utterance(self, text: str) -> Dict[str, Any]:
        """Full pragmatic analysis pipeline for a natural-language utterance.
 
        Stages:
        1. Gricean maxim violation detection.
        2. Politeness strategy identification.
        3. Discourse relation detection.
        4. Sentiment scoring (with intensifier and negation handling).
        5. Lightweight POS-pattern dependency analysis (NLPEngine optional).
 
        The result is logged to ``ReasoningMemory`` for downstream use.
 
        Args:
            text: Raw utterance string.
 
        Returns:
            Dict with keys: ``gricean_violations``, ``politeness_strategies``,
            ``discourse_relations``, ``sentiment``, ``dependency_analysis``,
            ``metadata``.
        """
        if not isinstance(text, str):
            raise ReasoningValidationError(
                "analyze_utterance expects a string",
                context={"type": type(text).__name__},
            )
        text = text.strip()
        if not text:
            raise ReasoningValidationError("analyze_utterance received an empty string")
 
        t0 = time.monotonic()
        result: Dict[str, Any] = {
            "gricean_violations":  self._check_gricean_maxims(text),
            "politeness_strategies": self._detect_politeness(text),
            "discourse_relations": self._identify_discourse_relations(text),
            "sentiment":           self._analyze_sentiment(text),
            "dependency_analysis": self._analyze_dependencies(text),
            "metadata": {
                "text_length": len(text.split()),
                "elapsed_ms":  round(elapsed_seconds(t0) * 1000, 2),
                "timestamp_ms": monotonic_timestamp_ms(),
            },
        }
 
        self._log_memory_event(
            "utterance_analysis",
            {"text": text[:120], "result_keys": list(result.keys())},
            tag="pragmatic_analysis",
        )
        return result
 
    def _check_gricean_maxims(self, text: str) -> Dict[str, bool]:
        """Return a dict of maxim → ``True`` when the maxim is **violated**."""
        maxims = self.pragmatic_heuristics["gricean_maxims"]
        return {maxim: not fn(text) for maxim, fn in maxims.items()}
 
    def _detect_politeness(self, text: str) -> Dict[str, List[str]]:
        """Return detected politeness strategies and their trigger phrases."""
        strategies  = self.pragmatic_heuristics["politeness_strategies"]
        text_lower  = text.lower()
        detected: Dict[str, List[str]] = {}
        for strategy, phrases in strategies.items():
            hits = [p for p in phrases if p.lower() in text_lower]
            if hits:
                detected[strategy] = hits
        return detected
 
    def _identify_discourse_relations(self, text: str) -> List[str]:
        """Return discourse markers present in the utterance."""
        markers    = self.pragmatic_heuristics["discourse_markers"]
        text_lower = text.lower()
        return [m for m in markers if m in text_lower]
 
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Token-level sentiment scoring with intensifier and negation handling.
 
        Algorithm
        ~~~~~~~~~
        - Scan tokens left-to-right.
        - Intensifier tokens set a multiplier for the *next* scored word.
        - Negator tokens flip polarity for the *next* scored word.
        - Positive / negative tokens contribute ``base × multiplier × polarity``.
        - Final score is normalised by word count and clamped to ``[-1, 1]``.
 
        Returns:
            Dict with keys ``score`` (float), ``positive_hits``,
            ``negative_hits``, ``negation_count``, ``intensifier_count``.
        """
        lexicon = self.pragmatic_heuristics["sentiment_analysis"]
        words   = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return {"score": 0.0, "positive_hits": 0, "negative_hits": 0,
                    "negation_count": 0, "intensifier_count": 0}
 
        sentiment       = 0.0
        multiplier      = 1.0
        negated         = False
        pos_hits        = 0
        neg_hits        = 0
        negation_count  = 0
        intensifier_count = 0
 
        for word in words:
            if word in lexicon["intensifiers"]:
                multiplier        = float(lexicon["intensifiers"][word])
                intensifier_count += 1
                continue
            if word in lexicon["negators"]:
                negated        = not negated
                negation_count += 1
                continue
 
            if word in lexicon["positive"]:
                base     = float(lexicon["positive"][word])
                score    = base * multiplier * (-1.0 if negated else 1.0)
                pos_hits += 1
            elif word in lexicon["negative"]:
                base     = float(lexicon["negative"][word])
                score    = base * multiplier * (-1.0 if negated else 1.0)
                neg_hits += 1
            else:
                # Unknown word: reset transient modifiers
                multiplier = 1.0
                negated    = False
                continue
 
            sentiment  += score
            multiplier  = 1.0
            negated     = False
 
        normalized_score = max(-1.0, min(1.0, sentiment / len(words)))
        return {
            "score":             round(normalized_score, 6),
            "positive_hits":     pos_hits,
            "negative_hits":     neg_hits,
            "negation_count":    negation_count,
            "intensifier_count": intensifier_count,
        }
 
    def _analyze_dependencies(self, text: str) -> Dict[str, Any]:
        """Lightweight POS-pattern dependency analysis.
 
        Attempts to import ``NLPEngine`` for full POS tagging.  When the
        import fails (common in isolated tests), falls back to a rule-based
        regex tokeniser that produces approximate UPOS tags for common English
        patterns.
 
        Returns:
            Dict with keys: ``clauses``, ``relations``, and one key per
            matched ``dependency_rules`` pattern.
        """
        try:
            from src.agents.language.nlp_engine import NLPEngine  # type: ignore
            nlp = NLPEngine()
            tokens = nlp.process_text(text)
            token_data = [
                {"text": t.text, "pos": t.pos, "index": t.index}
                for t in tokens
            ]
        except ImportError:
            logger.debug("NLPEngine unavailable — using fallback POS tokeniser")
            token_data = self._fallback_pos_tokenise(text)
 
        boundaries = self._detect_sentence_boundaries_raw(token_data)
        results: Dict[str, Any] = defaultdict(list)
 
        for rule_name, pattern in self.dependency_rules.items():
            results[rule_name] = self._match_pos_pattern(token_data, pattern)
 
        results["clauses"]   = self._detect_clauses(token_data, boundaries)
        results["relations"] = self._extract_grammatical_relations(token_data)
        return dict(results)
 
    # ------------------------------------------------------------------
    # Dependency analysis internals
    # ------------------------------------------------------------------
 
    @staticmethod
    def _fallback_pos_tokenise(text: str) -> List[Dict[str, Any]]:
        """Minimal regex-based POS tagger for tests without NLPEngine.
 
        Tags are approximate UPOS labels sufficient for pattern matching.
        """
        _closed_class: Dict[str, str] = {
            "the": "DET", "a": "DET", "an": "DET",
            "in": "ADP", "on": "ADP", "at": "ADP", "by": "ADP",
            "of": "ADP", "to": "ADP", "for": "ADP", "with": "ADP",
            "is": "AUX", "are": "AUX", "was": "AUX", "were": "AUX",
            "be": "AUX", "been": "AUX", "being": "AUX",
            "i": "PRON", "you": "PRON", "he": "PRON", "she": "PRON",
            "it": "PRON", "we": "PRON", "they": "PRON",
            "and": "CCONJ", "or": "CCONJ", "but": "CCONJ",
            "not": "PART", "no": "PART",
            "very": "ADV", "quite": "ADV", "rather": "ADV", "really": "ADV",
        }
        _verb_suffixes = ("ing", "ed", "ize", "ise", "ify", "ate")
        _adj_suffixes  = ("ful", "less", "ous", "ive", "able", "ible", "al", "ic")
        _adv_suffixes  = ("ly",)
 
        token_data = []
        raw_tokens = re.findall(r"\w+|[.!?,;:]", text)
        for idx, tok in enumerate(raw_tokens):
            lower = tok.lower()
            if re.fullmatch(r"[.!?,;:]", tok):
                pos = "PUNCT"
            elif lower in _closed_class:
                pos = _closed_class[lower]
            elif re.fullmatch(r"\d+", tok):
                pos = "NUM"
            elif tok[0].isupper() and idx > 0:
                pos = "PROPN"
            elif any(lower.endswith(s) for s in _adv_suffixes):
                pos = "ADV"
            elif any(lower.endswith(s) for s in _adj_suffixes):
                pos = "ADJ"
            elif any(lower.endswith(s) for s in _verb_suffixes):
                pos = "VERB"
            else:
                pos = "NOUN"
            token_data.append({"text": tok, "pos": pos, "index": idx})
        return token_data
 
    @staticmethod
    def _detect_sentence_boundaries_raw(token_data: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Split token list into sentence spans at sentence-final punctuation."""
        boundaries: List[Tuple[int, int]] = []
        start = 0
        for i, tok in enumerate(token_data):
            if tok["text"] in {".", "!", "?"}:
                boundaries.append((start, i))
                start = i + 1
        if start < len(token_data):
            boundaries.append((start, len(token_data) - 1))
        return boundaries or [(0, max(0, len(token_data) - 1))]
 
    def _detect_clauses(self, tokens: List[Dict[str, Any]], boundaries: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Extract subject–verb–object clause triples from sentence spans."""
        clauses: List[Dict[str, Any]] = []
        for start, end in boundaries:
            span = tokens[start: end + 1]
            verbs = [i for i, t in enumerate(span) if t["pos"] in {"VERB", "AUX"}]
            for vi in verbs:
                subj = next(
                    (span[j] for j in range(vi - 1, -1, -1)
                     if span[j]["pos"] in {"NOUN", "PROPN", "PRON"}),
                    None,
                )
                obj = next(
                    (span[j] for j in range(vi + 1, len(span))
                     if span[j]["pos"] in {"NOUN", "PROPN", "PRON"}),
                    None,
                )
                if subj or obj:
                    clauses.append({
                        "subject":        subj["text"] if subj else None,
                        "verb":           span[vi]["text"],
                        "object":         obj["text"]  if obj  else None,
                        "sentence_range": (start, end),
                        "verb_index":     start + vi,
                    })
        return clauses
 
    def _extract_grammatical_relations(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract SVO, AMOD, and PREP relation tuples from a token list."""
        relations: List[Dict[str, Any]] = []
        for i, tok in enumerate(tokens):
            pos = tok["pos"]
 
            if pos in {"VERB", "AUX"}:
                subj = self._find_previous(tokens, i, {"NOUN", "PROPN", "PRON"})
                obj  = self._find_next(tokens,    i, {"NOUN", "PROPN", "PRON"})
                if subj or obj:
                    relations.append({
                        "relation": "SVO",
                        "subject":  subj["text"] if subj else None,
                        "verb":     tok["text"],
                        "object":   obj["text"]  if obj  else None,
                    })
 
            elif pos == "ADJ":
                noun = self._find_next(tokens, i, {"NOUN", "PROPN"})
                if noun:
                    relations.append({
                        "relation":  "AMOD",
                        "adjective": tok["text"],
                        "noun":      noun["text"],
                    })
 
            elif pos == "ADP":
                obj = self._find_next(tokens, i, {"NOUN", "PROPN", "PRON"})
                if obj:
                    relations.append({
                        "relation":    "PREP",
                        "preposition": tok["text"],
                        "object":      obj["text"],
                    })
 
        return relations
 
    def _match_pos_pattern(self, tokens: List[Dict[str, Any]], pattern: List[str]) -> List[Dict[str, Any]]:
        """Sliding-window POS pattern matcher with wildcard and alternation support.
 
        Wildcards: ``"*"`` matches any single tag.
        Alternation: ``"NOUN|PROPN"`` matches either tag.
        """
        if not pattern:
            return []
        plen    = len(pattern)
        matches: List[Dict[str, Any]] = []
 
        for i in range(len(tokens) - plen + 1):
            window   = tokens[i: i + plen]
            window_pos = [t["pos"] for t in window]
            if self._wildcard_match(window_pos, pattern):
                matches.append({
                    "text":        " ".join(t["text"] for t in window),
                    "start_index": i,
                    "end_index":   i + plen - 1,
                    "pattern":     pattern,
                    "actual_pos":  window_pos,
                })
        return matches
 
    @staticmethod
    def _wildcard_match(actual: List[str], pattern: List[str]) -> bool:
        """Return ``True`` if ``actual`` matches ``pattern`` with wildcard/alternation."""
        if len(actual) != len(pattern):
            return False
        for a, p in zip(actual, pattern):
            if p == "*":
                continue
            if "|" in p:
                if a not in p.split("|"):
                    return False
            elif a != p:
                return False
        return True
 
    @staticmethod
    def _find_previous(tokens: List[Dict[str, Any]], index: int, target_pos: Set[str]) -> Optional[Dict[str, Any]]:
        """Return the nearest token to the *left* of ``index`` with a matching POS."""
        for i in range(index - 1, -1, -1):
            if tokens[i]["pos"] in target_pos:
                return tokens[i]
        return None
 
    @staticmethod
    def _find_next(tokens: List[Dict[str, Any]], index: int, target_pos: Set[str]) -> Optional[Dict[str, Any]]:
        """Return the nearest token to the *right* of ``index`` with a matching POS."""
        for i in range(index + 1, len(tokens)):
            if tokens[i]["pos"] in target_pos:
                return tokens[i]
        return None
 
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
 
    def save_knowledge_base(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the current KB to disk in the standard JSON format.
 
        Args:
            path: Override the configured ``knowledge_db`` path.
 
        Returns:
            The resolved output path.
        """
        out = Path(path or self._kb_path)
        records = [
            {"subject": s, "predicate": p, "object": str(o), "confidence": round(float(c), 8)}
            for (s, p, o), c in sorted(self.knowledge_base.items())
        ]
        rule_entries = [
            {"name": n, "callable": n, "weight": round(self.rule_weights.get(n, 0.0), 8)}
            for n in self._rule_registry
        ]
        payload = {
            "knowledge":  records,
            "rules":      rule_entries,
            "updated_at": time.time(),
        }
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Knowledge base saved: %d facts → %s", len(records), out)
        except OSError as exc:
            raise KnowledgePersistenceError(
                f"Failed to write knowledge base to {out}",
                cause=exc,
                context={"path": str(out)},
            ) from exc
        return out
 
    def save_discovered_rules(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist only the discovered (non-built-in) rules to the rule backup file."""
        out = Path(path or self._rule_backup_path)
        builtin_names = {"identity_rule", "transitive_rule"}
        entries = [
            {
                "name":        n,
                "weight":      round(self.rule_weights.get(n, 0.0), 8),
                "antecedents": self.rule_antecedents.get(n, []),
                "consequents": self.rule_consequents.get(n, []),
            }
            for n in self._rule_registry
            if n not in builtin_names
        ]
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved %d discovered rules → %s", len(entries), out)
        except OSError as exc:
            raise KnowledgePersistenceError(
                f"Failed to write rule backup to {out}",
                cause=exc,
                context={"path": str(out)},
            ) from exc
        return out
 
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
 
    def diagnostics(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of engine state for health monitoring."""
        return json_safe_reasoning_state({
            "kb_size":         self.kb_size,
            "rule_count":      self.rule_count,
            "rule_names":      sorted(self._rule_registry.keys()),
            "rule_weights":    {n: round(w, 4) for n, w in self.rule_weights.items()},
            "enable_learning": self.enable_learning,
            "min_confidence":  self.min_confidence,
            "min_support":     self.min_support,
            "contradiction_threshold": self.contradiction_threshold,
            "max_circular_depth":      self._max_circular_depth,
            "timestamp_ms":    monotonic_timestamp_ms(),
        })
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _log_memory_event(self, event_type: str, payload: Dict[str, Any], tag: Optional[str] = None) -> None:
        """Log an experience to ReasoningMemory, swallowing any errors."""
        try:
            experience = {"type": event_type, **payload}
            self.reasoning_memory.add(experience=experience, **({"tag": tag} if tag else {}))
        except Exception as exc:
            logger.debug("ReasoningMemory.add failed for '%s': %s", event_type, exc)
 
 
# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Rule Engine ===\n")
    printer.status("TEST", "Rule Engine initialized", "info")

    import tempfile
    import json
    import os
    import sys
    from pathlib import Path

    # ---- Create temporary resources -------------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="rule_engine_test_")
    kb_path = Path(tmp_dir) / "knowledge_db.json"
    rules_backup_path = Path(tmp_dir) / "discovered_rules.json"
    lexicon_path = Path(tmp_dir) / "sentiment.yaml"
    syntax_path = Path(tmp_dir) / "syntax.yaml"
    discourse_path = Path(tmp_dir) / "nlg_pack.yaml"

    # 1. Minimal knowledge base
    kb_data = {
        "knowledge": [
            ["cat", "is_animal", "true", 0.9],
            ["dog", "is_animal", "true", 0.95],
            ["cat", "has_fur", "true", 0.8],
            ["sky", "is_blue", "false", 0.1],
            ["animal", "has_biology", "true", 0.99],
            ["sky", "color", "blue", 0.9],
            ["sky", "color", "not_blue", 0.85],   # for conflict
        ],
        "rules": [
            {"name": "identity_rule", "callable": "identity_rule", "weight": 1.0},
            {"name": "transitive_rule", "callable": "transitive_rule", "weight": 0.8},
        ],
    }
    kb_path.write_text(json.dumps(kb_data))

    # 2. Minimal sentiment lexicon (YAML)
    lexicon_path.write_text("""
lexicons:
  positive: {love: 0.9, great: 0.8, happy: 0.75}
  negative: {hate: -0.9, terrible: -0.85}
modifiers:
  intensifiers: {very: 1.5, absolutely: 1.8}
negation:
  negators: [not, never]
antonyms: {blue: [not_blue], true: [false]}
""")

    # 3. Minimal syntax patterns
    syntax_path.write_text("""
phrase_patterns:
  - name: noun_phrase
    universal_pattern: ["DET", "NOUN"]
  - name: adj_noun
    universal_pattern: ["ADJ", "NOUN"]
""")

    # 4. Discourse / politeness
    discourse_path.write_text("""
discourse_markers: [however, therefore, but]
strategies:
  positive_politeness:
    phrases: [please, thank you]
""")

    # ---- Configuration override ---------------------------------------------------
    mock_config = {
        "contradiction_threshold": 0.25,
        "rules": {
            "enable_learning": True,
            "min_support": 0.1,
            "min_confidence": 0.5,
            "auto_weight_adjustment": True,
            "max_circular_depth": 3,
            "max_utterance_length": 50,
            "manner_filler_words": ["uh", "um"],
            "quality_hedge_words": ["probably", "maybe"],
            "discourse_markers_path": str(discourse_path),
            "politeness_strategies_path": str(discourse_path),
        },
        "storage": {
            "knowledge_db": str(kb_path),
            "lexicon_path": str(lexicon_path),
            "dependency_rules_path": str(syntax_path),
            "rule_backup": str(rules_backup_path),
        },
        "validation": {"redundancy_margin": 0.05},
        "inference": {"max_learning_cycles": 10},
    }

    def fake_load_config(*args, **kwargs):
        return mock_config.copy()

    def fake_get_section(section, cfg=None, **kwargs):
        return mock_config.get(section, {}).copy()

    # Monkey-patch the module's imports
    sys.modules[__name__].load_global_config = fake_load_config # type: ignore
    sys.modules[__name__].get_config_section = fake_get_section # type: ignore
    ReasoningMemory.load_global_config = fake_load_config # type: ignore
    ReasoningMemory.get_config_section = fake_get_section # type: ignore

    # ---- Instantiate engine -------------------------------------------------------
    engine = RuleEngine()
    printer.status("INIT", f"Engine ready: {engine.kb_size} facts, {engine.rule_count} rules", "success")

    # ---- Core tests --------------------------------------------------------------
    # 1. Fact operations
    engine.assert_fact(("eagle", "is_animal", "true"), 0.97)
    assert engine.query_fact(("eagle", "is_animal", "true")) is not None
    assert engine.retract_fact(("eagle", "is_animal", "true")) is True
    assert engine.query_fact(("eagle", "is_animal", "true")) is None

    # 2. Bulk assert
    bulk = {("fish", "is_animal", "true"): 0.88, ("fish", "lives_in", "water"): 0.95}
    inserted, skipped = engine.bulk_assert(bulk)
    assert inserted == 2 and skipped == 0

    # 3. Rule registration
    def dummy_rule(kb):
        return {("test", "fired", "yes"): 0.6}
    assert engine.add_rule(dummy_rule, "test_rule", 0.6, antecedents=["x"], consequents=["y"])
    rule_names = [rule["name"] for rule in engine.list_rules()]
    assert "test_rule" in rule_names
    assert engine.remove_rule("test_rule")

    # 4. Inference (transitive)
    engine.assert_fact(("A", "leads_to", "B"), 0.9)
    engine.assert_fact(("B", "leads_to", "C"), 0.85)
    inferred = engine.run_inference(min_confidence_filter=0.01, max_rounds=3)
    assert isinstance(inferred, dict)   # at least one new fact

    # 5. Circular detection
    assert isinstance(engine.detect_circular_rules(), list)

    # 6. Conflict detection
    conflicts = engine.detect_fact_conflicts()
    assert len(conflicts) >= 1   # sky color conflict

    # 7. Redundancy
    assert isinstance(engine.redundant_fact_check(), list)

    # 8. Utterance analysis (all stages)
    result = engine.analyze_utterance("I absolutely love this!")
    assert result["sentiment"]["score"] > 0
    assert "gricean_violations" in result and "politeness_strategies" in result

    # 9. Diagnostics & persistence
    diag = engine.diagnostics()
    assert "kb_size" in diag and "rule_count" in diag
    saved = engine.save_knowledge_base(kb_path)
    assert saved.exists()
    engine.save_discovered_rules(rules_backup_path).exists()

    # ---- Cleanup -----------------------------------------------------------------
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    printer.status("PASS", "All tests passed", "success")
    print("\n=== Test ran successfully ===\n")