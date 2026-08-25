"""
Rule Engine – fixed sector inference, idempotent rule loading, and config-controlled behavior.
"""

import time
import os
import json
import re
import traceback
import warnings

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from .inference_result import InferenceResult, InferenceTrace
from ..utils.knowledge_errors import *
from ..utils.knowledge_helpers import *
from ..utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Rule Engine")
printer = PrettyPrinter()


# -----------------------------------------------------------------------------
# Worker function – runs in a separate process (must be at module level for pickling)
# -----------------------------------------------------------------------------
def _run_rule_code(rule_code: str, knowledge_base: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Execute rule code in a restricted environment.
    Returns inferred facts dict.
    """
    safe_globals = {
        '__builtins__': {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'dict': dict, 'enumerate': enumerate, 'float': float,
            'int': int, 'len': len, 'list': list, 'max': max,
            'min': min, 'range': range, 'round': round, 'set': set,
            'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
            'True': True, 'False': False, 'None': None,
        },
        'kb': knowledge_base,
    }
    local_ns = {'inferred': {}}
    exec(rule_code, safe_globals, local_ns)
    result = local_ns.get('inferred', {})
    if not isinstance(result, dict):
        raise TypeError(f"Rule returned {type(result)}, expected dict")
    return result


# -----------------------------------------------------------------------------
# Rule Engine - Production Ready (no daemonic nesting)
# -----------------------------------------------------------------------------
class RuleEngine:

    SECTORS = {"civic", "medical", "economic", "scientific", "philosophical", "technological"}

    # Sector keywords for inference (moved to class constant)
    SECTOR_KEYWORDS = {
        "civic": {"law", "governance", "citizen", "vote", "policy", "courtroom", "privacy"},
        "medical": {"diagnosis", "symptom", "treatment", "patient", "disease", "medicine"},
        "economic": {"market", "inflation", "currency", "gdp", "employment", "money", "dollar", "recession"},
        "scientific": {"experiment", "hypothesis", "equation", "physics", "biology", "research"},
        "philosophical": {"ethics", "metaphysics", "ontology", "logic", "thought", "consciousness"},
        "technological": {"ai", "robot", "algorithm", "neural", "software", "agents", "data", "code"},
    }

    def __init__(self):
        self.config = load_global_config()
        self.enabled = self.config.get('enabled', True)

        self.rule_config = get_config_section('rule_engine')
        self.verbose_logging = self.rule_config.get('verbose_logging', False)
        self.auto_discover = self.rule_config.get('auto_discover', True)
        self.min_rule_confidence = self.rule_config.get('min_rule_confidence', 0.6)
        self.slow_rule_threshold = self.rule_config.get('slow_rule_threshold', 0.5)
        self.rule_timeout = self.rule_config.get('rule_timeout_seconds', 1.0)
        self.max_concurrent_rules = self.rule_config.get('max_concurrent_rules', 4)
        self.rule_sources = self.rule_config.get('rule_sources', [])
        # save_inferred is deprecated – persistence belongs to orchestration
        self.save_inferred = self.rule_config.get('save_inferred', False)
        if self.save_inferred:
            warnings.warn("'save_inferred' is deprecated and ignored. Persist inference results at the orchestration level.", DeprecationWarning)
        self.rules_dir = self.rule_config.get('rules_dir', 'src/agents/knowledge/templates/')
        self.max_facts_per_rule = self.rule_config.get('max_facts_per_rule', 10)

        # Internal data structures – split source vs runtime
        self._source_rules: List[Dict] = []
        self._runtime_rules: List[Dict] = []
        # Combined view (computed property)
        self._rules_combined: List[Dict] = []
        self.sector_rules: Dict[str, List[Dict]] = defaultdict(list)
        self.category_rules: Dict[str, List[Dict]] = defaultdict(list)

        # Failure tracking
        self.rule_failure_counts: Dict[str, int] = defaultdict(int)
        self.rule_timeout_counts: Dict[str, int] = defaultdict(int)
        self.rule_last_error: Dict[str, str] = {}

        # Process pool (non‑daemonic workers)
        self._executor = None

        # Load rules only if enabled
        if self.enabled:
            self.reload_rules()
        else:
            logger.info("RuleEngine is disabled – no rules loaded.")

    @property
    def rules(self) -> List[Dict]:
        """Combined list of source-loaded and runtime-added rules."""
        return self._rules_combined

    def _rebuild_combined(self):
        """Rebuild the combined rule list and indices from source + runtime."""
        self._rules_combined = self._source_rules + self._runtime_rules
        # Rebuild indices
        self.sector_rules.clear()
        self.category_rules.clear()
        for rule in self._rules_combined:
            for tag in rule["tags"]:
                tag_lower = tag.lower()
                if tag_lower in self.SECTORS:
                    self.sector_rules[tag_lower].append(rule)
                self.category_rules[tag_lower].append(rule)

    # ----------------------------------------------------------------------
    # Rule loading / reloading (idempotent)
    # ----------------------------------------------------------------------
    def reload_rules(self) -> None:
        """
        Reload source rules from configured sources (explicit + auto-discovered).
        Runtime-added rules are preserved.
        """
        if not self.enabled:
            logger.warning("RuleEngine is disabled – reload_rules() has no effect.")
            return

        # Clear source rules only
        self._source_rules.clear()
        loaded_files = set()

        # 1. Load explicit rule_sources (if any)
        for source_path in self.rule_sources:
            if not source_path:
                continue
            # Resolve relative path from project root if needed
            path = self._resolve_path(source_path)
            if path.exists() and path.suffix.lower() in ('.json',):
                self._load_rules_from_file(str(path), source_type="explicit")
                loaded_files.add(str(path))
            else:
                logger.warning(f"Explicit rule source not found: {path}")

        # 2. Auto-discover from rules_dir if enabled
        if self.auto_discover:
            if os.path.isdir(self.rules_dir):
                for filename in os.listdir(self.rules_dir):
                    if filename.endswith("_rules.json"):
                        file_path = os.path.join(self.rules_dir, filename)
                        if file_path not in loaded_files:  # avoid duplicate loading
                            self._load_rules_from_file(file_path, source_type="discovered")
                            loaded_files.add(file_path)
            else:
                logger.warning(f"Rules directory not found: {self.rules_dir}")

        # Rebuild combined list and indices
        self._rebuild_combined()
        logger.info(f"Reloaded {len(self._source_rules)} source rules, "
                    f"{len(self._runtime_rules)} runtime rules preserved.")

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path relative to project root if not absolute."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        # Assume relative to project root (which is 4 levels up from this file)
        base = Path(__file__).parent.parent.parent.parent.parent
        return (base / path).resolve()

    def _load_rules_from_file(self, file_path: str, source_type: str = "unknown"):
        """Load rules from a JSON file and append to _source_rules."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                rules_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load rules from {file_path}: {e}")
            return

        if not isinstance(rules_data, list):
            logger.warning(f"Skipping {file_path}: root must be a list of rule definitions.")
            return

        for rule_def in rules_data:
            if "implementation" not in rule_def:
                note = rule_def.get("symbolic_ai_implementation_note")
                if note:
                    rule_def["implementation"] = f"# symbolic placeholder\n# {note}"
                else:
                    logger.warning(f"Skipping rule with missing implementation: {rule_def.get('name', 'Unnamed')}")
                    continue

            name = rule_def["name"]
            code = rule_def["implementation"]
            weight = rule_def.get("weight", 1.0)
            tags = rule_def.get("tags", [])
            metadata = {
                "description": rule_def.get("description", ""),
                "source": rule_def.get("source", file_path),
                "type": rule_def.get("type", "heuristic")
            }

            # Check for duplicate names in source rules – overwrite? We'll replace.
            existing_index = next((i for i, r in enumerate(self._source_rules) if r["name"] == name), None)
            rule_entry = {
                "name": name,
                "code": code,
                "weight": weight,
                "tags": tags or [],
                "meta": metadata,
                "_source": file_path,
                "_source_type": source_type,
            }
            if existing_index is not None:
                self._source_rules[existing_index] = rule_entry
                logger.debug(f"Replaced source rule '{name}' from {file_path}")
            else:
                self._source_rules.append(rule_entry)
                logger.debug(f"Loaded source rule '{name}' from {file_path}")

    # ----------------------------------------------------------------------
    # Public rule management (runtime rules)
    # ----------------------------------------------------------------------
    def add_rule(self,
                 name: str,
                 rule_code: str,
                 weight: float = 1.0,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None) -> None:
        """Add a runtime rule. Idempotent: replaces if name already exists."""
        if not self.enabled:
            raise RuntimeError("RuleEngine is disabled; cannot add rules.")
        if weight <= 0:
            raise ValueError("Weight must be positive")
        if not isinstance(rule_code, str):
            raise TypeError("rule_code must be a string")

        rule = {
            "name": name,
            "code": rule_code,
            "weight": weight,
            "tags": tags or [],
            "meta": metadata or {},
            "_source": "runtime",
            "_source_type": "runtime",
        }
        # Replace existing runtime rule with same name
        existing_index = next((i for i, r in enumerate(self._runtime_rules) if r["name"] == name), None)
        if existing_index is not None:
            self._runtime_rules[existing_index] = rule
            logger.debug(f"Replaced runtime rule '{name}'")
        else:
            self._runtime_rules.append(rule)
            logger.debug(f"Added runtime rule '{name}'")

        self._rebuild_combined()

    def remove_rule(self, name: str) -> bool:
        """Remove a runtime rule by name. Returns True if removed."""
        for i, rule in enumerate(self._runtime_rules):
            if rule["name"] == name:
                self._runtime_rules.pop(i)
                self._rebuild_combined()
                logger.info(f"Removed runtime rule: {name}")
                return True
        logger.warning(f"Rule '{name}' not found in runtime rules; cannot remove.")
        return False

    # ----------------------------------------------------------------------
    # Backward compatibility – deprecated but retained
    # ----------------------------------------------------------------------
    def load_all_sectors(self):
        """
        Deprecated. Use reload_rules() instead.
        Kept for backward compatibility; calls reload_rules().
        """
        warnings.warn("load_all_sectors() is deprecated; use reload_rules() instead.", DeprecationWarning, stacklevel=2)
        self.reload_rules()

    def load_rules_from_json(self, path: str):
        """
        Deprecated. Use reload_rules() and rule_sources config instead.
        Kept for backward compatibility; loads as source rules.
        """
        warnings.warn("load_rules_from_json() is deprecated; add file to rule_sources or use add_rule().", DeprecationWarning, stacklevel=2)
        self._load_rules_from_file(path, source_type="explicit")
        self._rebuild_combined()

    def get_rules_by_category(self, category: str) -> List[Dict]:
        return self.category_rules.get(category.lower(), [])

    # ----------------------------------------------------------------------
    # Recursive term extraction for arbitrary knowledge base shapes
    # ----------------------------------------------------------------------
    @staticmethod
    def _iter_terms(value: Any):
        """Recursively yield all alphabetic tokens from nested structures."""
        if isinstance(value, dict):
            for key, item in value.items():
                yield from RuleEngine._iter_terms(key)
                yield from RuleEngine._iter_terms(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                yield from RuleEngine._iter_terms(item)
            return
        if value is None:
            return
        for term in re.findall(r"[a-zA-Z]+", str(value).lower()):
            yield term

    def _select_smart_rules(
        self,
        knowledge_base: Mapping[Any, Any],
    ) -> Tuple[List[Dict], str, Optional[str]]:
        """Select the rule set used by smart inference.
    
        Returns:
            rules:
                Rules that should be executed.
    
            detected_sector:
                Sector inferred from the knowledge base.
    
            execution_sector:
                Sector under which the selected rules are actually scoped.
                ``None`` means the complete rule set is being used.
        """
        detected_sector = self.infer_sector(dict(knowledge_base))
    
        if (
            detected_sector != "general"
            and self.sector_rules.get(detected_sector)
        ):
            return (
                self.sector_rules[detected_sector],
                detected_sector,
                detected_sector,
            )
    
        return self.rules, detected_sector, None

    def infer_sector(self, knowledge_base: dict) -> str:
        """Determine the most likely sector based on keywords in the knowledge base."""
        all_terms = set(self._iter_terms(knowledge_base))
        sector_scores = {}
        for sector, terms in self.SECTOR_KEYWORDS.items():
            sector_scores[sector] = len(all_terms.intersection(terms))
        best_sector = max(sector_scores, key=lambda s: sector_scores[s])
        if sector_scores[best_sector] == 0:
            return "general"
        return best_sector

    # ----------------------------------------------------------------------
    # Rule application methods
    # ----------------------------------------------------------------------
    def _check_enabled(self):
        if not self.enabled:
            raise RuntimeError("RuleEngine is disabled. Cannot apply rules.")

    def infer(self, knowledge_base: Mapping[Any, Any], *, sector: Optional[str] = None, smart: bool = True, trace: bool = False) -> InferenceResult:
        """Run inference and return the canonical typed result.
    
        Args:
            knowledge_base:
                Structured knowledge supplied to rule execution.
    
            sector:
                Optional explicit sector. When supplied, only rules indexed
                under that sector are considered.
    
            smart:
                When True and no explicit sector is supplied, automatically
                select the most relevant sector. When False, execute the
                complete rule set.
    
            trace:
                When True, retain provenance for every accepted rule
                contribution.
    
        Returns:
            A stable InferenceResult.
    
        Notes:
            This method performs inference only. It does not persist results
            into KnowledgeMemory. Persistence belongs to orchestration.
        """
        self._check_enabled()
    
        kb = dict(knowledge_base)
    
        result_sector: Optional[str] = None
        execution_sector: Optional[str] = None
    
        if sector is not None:
            normalized_sector = sector.lower()
            rules_to_use = self.sector_rules.get(normalized_sector, [])
            result_sector = normalized_sector
            execution_sector = normalized_sector
    
        elif smart:
            (rules_to_use, result_sector, execution_sector) = self._select_smart_rules(kb)
    
        else:
            rules_to_use = self.rules
    
        raw_result = self._apply_rules(
            rules_to_use,
            kb,
            verbose=False,
            sector_tag=execution_sector,
            collect_all_traces=trace,
        )
    
        if trace:
            facts, raw_traces = raw_result
    
            traces = [
                InferenceTrace(
                    fact=item["fact"],
                    confidence=float(item["confidence"]),
                    rule=str(item["rule"]),
                    source=str(
                        item.get(
                            "source",
                            "unknown",
                        )
                    ),
                    sector=item.get("sector"),
                )
                for item in raw_traces
            ]
    
        else:
            facts = raw_result
            traces = []
    
        return InferenceResult(facts=dict(facts), traces=traces, sector=result_sector) # type: ignore

    def smart_apply(self, knowledge_base: Mapping[Any, Any], verbose: bool = False) -> Union[dict, Tuple[dict, list]]:
        """Applies rules from the most relevant sector, or all if none matches."""
        self._check_enabled()
        printer.status("ENGINE", "Applying rules", "info")
    
        rules_to_use, detected_sector, execution_sector = (self._select_smart_rules(knowledge_base))
    
        if execution_sector is not None:
            logger.info("[RuleEngine] Smart Apply: Using sector-specific rules for: %s", detected_sector)
        else:
            logger.info(
                "[RuleEngine] Smart Apply: No specific rule set available "
                "for sector '%s'; applying all rules.",
                detected_sector,
            )
    
        return self._apply_rules(
            rules_to_use,
            knowledge_base,
            verbose=verbose,
            sector_tag=execution_sector,
        )

    def apply(self, knowledge_base: Mapping[Any, Any], verbose: bool = False) -> Union[dict, Tuple[dict, list]]:
        """Apply all rules in the engine."""
        self._check_enabled()
    
        return self._apply_rules(self.rules, knowledge_base, verbose=verbose)

    def apply_by_sector(self, knowledge_base: Mapping[Any, Any], sector: str, verbose: bool = False) -> Union[dict, Tuple[dict, list]]:
        """Apply rules belonging to a specific sector."""
        self._check_enabled()
        printer.status("ENGINE", "Applying by sectors", "info")
        rules = self.sector_rules.get(sector.lower(), [])
        return self._apply_rules(rules, knowledge_base, verbose=verbose, sector_tag=sector)

    def _apply_rules(self, rules: List[Dict], knowledge_base: Mapping[Any, Any], verbose: bool = False,
        sector_tag: Optional[str] = None, collect_all_traces: bool = False) -> Union[dict, Tuple[dict, list]]:
        """Execute a list of rules with sandboxing, timeouts, and failure tracking."""
        inferred: Dict[Any, float] = {}
        traces: List[Dict[str, Any]] = []
        executor = self._get_executor()
        # Normalize Mapping inputs before crossing the process boundary.
        kb = dict(knowledge_base)
        futures = []
        
        for rule in rules:
            future = executor.submit(_run_rule_code, rule["code"], kb)
            futures.append((rule, future, time.perf_counter()))

        for rule, future, start_time in futures:
            try:
                results = future.result(timeout=self.rule_timeout)
                exec_time = time.perf_counter() - start_time
                if exec_time > self.slow_rule_threshold:
                    logger.debug(f"Slow rule {rule['name']}: {exec_time:.2f}s")

                for fact, confidence in results.items():
                    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                        raise TypeError(
                            f"Rule '{rule['name']}' produced a non-numeric "
                            f"confidence for fact {fact!r}: {confidence!r}"
                        )
                
                    weighted_confidence = (float(confidence) * float(rule["weight"]))
                
                    if weighted_confidence < self.min_rule_confidence:
                        continue
                
                    trace_entry = {
                        "fact": fact,
                        "confidence": weighted_confidence,
                        "rule": rule["name"],
                        "source": rule["meta"].get(
                            "source",
                            "unknown",
                        ),
                        "sector": sector_tag,
                    }
                
                    # Canonical typed inference can retain every accepted
                    # rule contribution for provenance/support analysis.
                    if collect_all_traces:
                        traces.append(trace_entry)
                
                    # Final facts continue to use the strongest accepted
                    # confidence, preserving existing inference semantics.
                    if (
                        fact not in inferred
                        or weighted_confidence > inferred[fact]
                    ):
                        inferred[fact] = weighted_confidence
                
                        # Preserve existing verbose behavior for legacy callers.
                        if (
                            (verbose or self.verbose_logging)
                            and not collect_all_traces
                        ):
                            traces.append(trace_entry)

                self.rule_failure_counts[rule["name"]] = 0

            except FutureTimeoutError:
                self.rule_timeout_counts[rule["name"]] += 1
                self.rule_failure_counts[rule["name"]] += 1
                self.rule_last_error[rule["name"]] = f"Timeout after {self.rule_timeout}s"
                logger.warning(f"[RuleEngine] Rule {rule['name']} timed out after {self.rule_timeout}s")
                future.cancel()

            except Exception as e:
                self.rule_failure_counts[rule["name"]] += 1
                self.rule_last_error[rule["name"]] = f"{type(e).__name__}: {e}"
                logger.warning(f"[RuleEngine] Rule {rule['name']} failed: {e}\n{traceback.format_exc()}")

        if self.max_facts_per_rule > 0 and len(inferred) > self.max_facts_per_rule:
            inferred = dict(sorted(inferred.items(), key=lambda x: x[1], reverse=True)[:self.max_facts_per_rule])

        if verbose or collect_all_traces:
            return inferred, traces

        return inferred

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_concurrent_rules)
        return self._executor

    # ----------------------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------------------
    def save_rules(self, path: str):
        """Export rule metadata (not code) to JSON for inspection."""
        with open(path, "w") as f:
            json.dump([{
                "name": r["name"],
                "weight": r["weight"],
                "tags": r["tags"],
                "meta": r["meta"],
                "source": r.get("_source", "unknown")
            } for r in self.rules], f, indent=2)

    def get_failure_stats(self) -> Dict[str, Any]:
        return {
            "failures": dict(self.rule_failure_counts),
            "timeouts": dict(self.rule_timeout_counts),
            "last_errors": self.rule_last_error
        }

    def close(self):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        self.close()


# -----------------------------------------------------------------------------
# Expanded RuleService with caching, management, and error handling
# -----------------------------------------------------------------------------
class RuleService:
    """High-level service for rule inference with caching and management."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config_section('rule_engine')
        self.engine = RuleEngine()
        self.logger = get_logger("RuleService")

        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_max_size = self.config.get('cache_max_size', 100)
        self.cache_ttl = self.config.get('cache_ttl_seconds', 300)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._failure_callback: Optional[Callable[[Dict], None]] = None

    def _check_enabled(self):
        if not self.engine.enabled:
            raise RuntimeError("RuleService is disabled because the underlying RuleEngine is disabled.")

    def _handle_failure(self, error_info: Dict):
        logger.error(f"RuleService failure: {error_info}")
        if self._failure_callback:
            try:
                self._failure_callback(error_info)
            except Exception as e:
                logger.error(f"Failure callback raised an exception: {e}")

    def infer(self, knowledge_base: Mapping[Any, Any],
              sector: Optional[str] = None,
              smart: bool = True,
              trace: bool = False,
              use_cache: bool = True) -> InferenceResult:
        """Return the canonical typed inference result."""
        self._check_enabled()
        cache_key = None
    
        if use_cache and self.cache_enabled:
            cache_key = safe_hash(
                {
                    "knowledge_base": dict(knowledge_base),
                    "sector": sector,
                    "smart": smart,
                    "trace": trace,
                }
            )
    
        if cache_key and cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
    
            if (time.time() - timestamp) < self.cache_ttl:
                self.logger.debug(
                    "Inference cache hit: %s",
                    cache_key[:8],
                )
                return cached_result
    
            self._cache.pop(cache_key, None)
    
        result = self.engine.infer(
            knowledge_base,
            sector=sector,
            smart=smart,
            trace=trace,
        )
    
        if use_cache and self.cache_enabled and cache_key:
            if len(self._cache) >= self.cache_max_size:
                oldest_key = min(
                    self._cache,
                    key=lambda key: self._cache[key][1],
                )
                del self._cache[oldest_key]
    
            self._cache[cache_key] = (
                result,
                time.time(),
            )

        return result

    def query(self, knowledge_base: Dict[str, Any], sector: Optional[str] = None, verbose: bool = False,
              use_cache: bool = True) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict]]]:
        """Backward-compatible dictionary-based query API."""
        try:
            typed_result = self.infer(
                knowledge_base,
                sector=sector,
                smart=sector is None,
                trace=verbose,
                use_cache=use_cache,
            )

        except Exception as exc:
            logger.error(f"Rule inference failed: {exc}", exc_info=True)

            raise KnowledgeError(
                error_type=KnowledgeErrorType.RULE_TIMEOUT,
                message=f"Rule inference failed: {exc}",
                severity=Severity.HIGH,
                context={
                    "sector": sector,
                    "verbose": verbose,
                },
            ) from exc

        if not verbose:
            return typed_result.facts

        legacy_traces = [
            {
                "fact": item.fact,
                "confidence": item.confidence,
                "rule": item.rule,
                "source": item.source,
                "sector": item.sector,
            }
            for item in typed_result.traces
        ]

        return typed_result.facts, legacy_traces

    def apply(self, knowledge_base: Mapping[Any, Any], verbose: bool = False) -> Any:
        return self.engine.apply(knowledge_base, verbose=verbose)

    def smart_apply(self, knowledge_base: Mapping[Any, Any], verbose: bool = False) -> Any:
        return self.engine.smart_apply(knowledge_base, verbose=verbose)


    def load_all_sectors(self) -> None:
        self.engine.load_all_sectors()
        self.clear_cache()

    def add_rule(self, name: str, rule_code: str, weight: float = 1.0,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None) -> None:
        self.engine.add_rule(name, rule_code, weight, tags, metadata)
        self.clear_cache()

    def remove_rule(self, name: str) -> bool:
        removed = self.engine.remove_rule(name)
        if removed:
            self.clear_cache()
        return removed

    def reload_rules(self) -> None:
        """Reload source rules (preserving runtime rules)."""
        self.engine.reload_rules()
        self.clear_cache()

    def list_rules(self) -> List[Dict]:
        return [{
            "name": r["name"],
            "weight": r["weight"],
            "tags": r["tags"],
            "meta": r["meta"],
            "source": r.get("_source", "unknown")
        } for r in self.engine.rules]

    def get_stats(self) -> Dict[str, Any]:
        stats = self.engine.get_failure_stats()
        stats["cache"] = {
            "enabled": self.cache_enabled,
            "size": len(self._cache),
            "max_size": self.cache_max_size,
            "ttl_seconds": self.cache_ttl
        }
        return stats

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.cache_max_size,
            "ttl_seconds": self.cache_ttl,
            "enabled": self.cache_enabled,
        }

    def set_failure_callback(self, callback: Callable[[Dict], None]) -> None:
        self._failure_callback = callback

    def close(self) -> None:
        self.engine.close()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    "RuleEngine",
    "RuleService",
]


# -----------------------------------------------------------------------------
# Self-test / demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Rule Engine (idempotent, config-controlled) ===\n")
    engine = RuleEngine()
    service = RuleService()

    # Test with different knowledge base shapes
    test_bases = [
        {("AI", "is_a", "Technology"): 1.0, ("Socrates", "is_a", "Human"): 0.95},
        {"AI": "Technology", "Socrates": "Human"},
        {"data": {"AI": {"is_a": "Technology"}}},
    ]

    for kb in test_bases:
        sector = engine.infer_sector(kb)
        print(f"KB: {kb} -> inferred sector: {sector}")
        result = engine.smart_apply(kb, verbose=False)
        printer.pretty("SMART APPLY", result, "success" if result else "error")

    # Test idempotent reload
    print("\n--- Reloading rules (should not duplicate) ---")
    before_count = len(engine.rules)
    engine.reload_rules()
    after_count = len(engine.rules)
    print(f"Rules before: {before_count}, after: {after_count} (should be equal)")

    # Test runtime rule addition
    engine.add_rule("test_rule", "inferred['test'] = 0.9", weight=0.8, tags=["test"])
    print(f"Rules after adding runtime rule: {len(engine.rules)}")
    engine.remove_rule("test_rule")
    print(f"Rules after removing runtime rule: {len(engine.rules)}")

    engine.close()
    print("\n=== Success ===\n")
