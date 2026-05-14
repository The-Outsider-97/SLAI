"""
Factory/policy layer for single and combined reasoning strategies.

Provides:
- Dynamic discovery and creation of reasoning strategies.
- Instance caching (optional) to avoid re‑instantiating common strategies.
- Combined reasoning with sequential execution and context passing.
- Keyword‑based reasoning strategy suggestion.
- Full integration with ReasoningMemory and ReasoningCache.
"""
from __future__ import annotations

import inspect
import threading
from typing import Any, Dict, List, Optional, Type, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from .types import *
from .reasoning_memory import ReasoningMemory
from .reasoning_cache import ReasoningCache
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Types")
printer = PrettyPrinter()

class ReasoningTypes:
    """
    Factory and policy layer for reasoning strategies.

    Features:
    - Create single reasoning strategies (e.g., "abduction").
    - Create combined strategies using "+" syntax (e.g., "abduction+induction+deduction").
    - Optional caching of created instances to reduce overhead.
    - Keyword‑based suggestion of suitable reasoning strategy for a problem.
    - Shared memory and cache for all reasoning instances.
    """

    # Core mapping – can be extended by discovery or configuration
    _TASK_TYPES: Dict[str, Type[BaseReasoning]] = {
        "abduction": ReasoningAbduction,
        "deduction": ReasoningDeductive,
        "induction": ReasoningInductive,
        "analogical": ReasoningAnalogical,
        # "analitical": ReasoningAnalogical,      # backward‑compatible typo
        "decompositional": ReasoningDecompositional,
        "cause_effect": ReasoningCauseAndEffect,
    }

    def __init__(self) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.types_cfg: Dict[str, Any] = get_config_section("reasoning_types", self.config) or {}

        # ---- Configuration -------------------------------------------------
        self.max_combined_types: int = bounded_iterations(
            self.types_cfg.get("max_combined_types", 3), minimum=1, maximum=10
        )
        self.enable_instance_cache: bool = bool(self.types_cfg.get("enable_instance_cache", False))
        self.instance_cache_max_size: int = bounded_iterations(
            self.types_cfg.get("instance_cache_max_size", 32), minimum=1, maximum=10000
        )
        self.instance_cache_ttl_seconds: Optional[float] = self._optional_float(
            self.types_cfg.get("instance_cache_ttl_seconds", 300.0)
        )
        self.default_strategy: str = str(self.types_cfg.get("default_strategy", "abduction+deduction")).strip()
        self.strategy_keywords: Dict[str, List[str]] = self.types_cfg.get("strategy_keywords", {
            "abduction": ["explain", "why", "hypothesis", "plausible", "most likely"],
            "deduction": ["prove", "derive", "therefore", "premise", "must be"],
            "induction": ["pattern", "trend", "generalize", "predict", "extrapolate"],
            "analogical": ["analogy", "similar", "compare", "resembles"],
            "decompositional": ["break down", "component", "decompose", "subsystem"],
            "cause_effect": ["effect", "impact", "results in", "causal", "cause"],
        })

        # ---- Shared resources ----------------------------------------------
        self.reasoning_memory: ReasoningMemory = ReasoningMemory()
        self._instance_cache: Optional[ReasoningCache] = None
        if self.enable_instance_cache:
            self._instance_cache = ReasoningCache(
                namespace="reasoning_types_instances",
                max_size=self.instance_cache_max_size,
                default_ttl_seconds=self.instance_cache_ttl_seconds,
                memory=self.reasoning_memory if self.types_cfg.get("record_memory_events", False) else None,
            )

        self._lock = threading.RLock()
        self._stats: Dict[str, int] = {"single": 0, "combined": 0}
        logger.info(
            "ReasoningTypes initialized | cache_enabled=%s | max_combined=%s",
            self.enable_instance_cache, self.max_combined_types
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create(self, task_type: str) -> BaseReasoning:
        """
        Create a reasoning strategy instance (single or combined).

        Args:
            task_type: One of the registered strategy names, or a combination
                       joined by '+' (e.g., "abduction+induction+deduction").

        Returns:
            An instance of BaseReasoning (single or combined).

        Raises:
            ReasoningTypeError: If the task_type is unknown or invalid.
        """
        normalized = (task_type or "").strip().lower()
        if not normalized:
            raise ReasoningTypeError("reasoning type is required", context={"task_type": task_type})

        # Check cache first (if enabled)
        if self.enable_instance_cache and self._instance_cache is not None:
            cached = self._instance_cache.get(normalized)
            if cached is not None:
                logger.debug(f"Returning cached reasoning instance: {normalized}")
                return cached

        # Combined reasoning
        if "+" in normalized:
            instance = self._create_combined_reasoning(normalized)
        else:
            instance = self._create_single_reasoning(normalized)

        # Store in cache (if enabled) – note: combined instances are also cached
        if self.enable_instance_cache and self._instance_cache is not None:
            self._instance_cache.set(normalized, instance, metadata={"type": "combined" if "+" in normalized else "single"})

        return instance

    def determine_reasoning_strategy(self, problem: str) -> str:
        """
        Suggest a reasoning strategy based on keyword matching.

        Args:
            problem: Natural language description of the problem.

        Returns:
            A reasoning type string (e.g., "abduction") or a combined type
            if multiple keywords match; otherwise the default strategy.
        """
        text = (problem or "").lower()
        matched = set()
        for strategy, keywords in self.strategy_keywords.items():
            if any(kw in text for kw in keywords):
                matched.add(strategy)

        if not matched:
            return self.default_strategy

        # If more than one strategy matches, combine them (up to max_combined_types)
        sorted_matches = sorted(matched)  # deterministic order
        if len(sorted_matches) > self.max_combined_types:
            sorted_matches = sorted_matches[:self.max_combined_types]
        return "+".join(sorted_matches)

    def get_memory(self) -> ReasoningMemory:
        """Return the shared reasoning memory instance."""
        return self.reasoning_memory

    def get_cache(self) -> Optional[ReasoningCache]:
        """Return the optional instance cache, if enabled."""
        return self._instance_cache
    
    def register(self, name: str, cls: Type[BaseReasoning]) -> None:
        normalized = name.strip().lower()
        if not normalized or not issubclass(cls, BaseReasoning):
            raise ReasoningTypeError("Invalid reasoning type registration")
        with self._lock:
            self._TASK_TYPES[normalized] = cls

    def discover_task_types(self) -> None:
        """
        Dynamically discover all BaseReasoning subclasses in the types module
        and register them under their normalized names.
        """
        import src.agents.reasoning.types as reasoning_types_module # type: ignore

        with self._lock:
            for _, obj in inspect.getmembers(reasoning_types_module):
                if (inspect.isclass(obj) and issubclass(obj, BaseReasoning)
                        and obj is not BaseReasoning):
                    name = obj.__name__.replace("Reasoning", "").lower()
                    if name not in self._TASK_TYPES:
                        self._TASK_TYPES[name] = obj
                        logger.debug(f"Discovered reasoning type: {name} -> {obj.__name__}")

    # ------------------------------------------------------------------
    # Internal factories
    # ------------------------------------------------------------------
    def _create_single_reasoning(self, normalized: str) -> BaseReasoning:
        """Instantiate a single reasoning strategy."""
        with self._lock:
            reasoning_cls = self._TASK_TYPES.get(normalized)
            if reasoning_cls is None:
                self.discover_task_types()
                reasoning_cls = self._TASK_TYPES.get(normalized)

        if reasoning_cls is None:
            raise ReasoningTypeError(
                f"Unknown reasoning type: {normalized}",
                context={"available": sorted(self._TASK_TYPES.keys())}
            )
        return reasoning_cls()

    def _create_combined_reasoning(self, combined_type: str) -> BaseReasoning:
        """
        Create a combined reasoning instance that runs multiple strategies
        sequentially, passing context and results forward.
        """
        names = [n.strip() for n in combined_type.split("+") if n.strip()]
        if not (1 <= len(names) <= self.max_combined_types):
            raise ReasoningTypeError(
                f"Combined reasoning must include between 1 and {self.max_combined_types} strategies",
                context={"requested": len(names), "max": self.max_combined_types}
            )

        # Resolve each component class
        components: List[BaseReasoning] = []
        component_names: List[str] = []
        for name in names:
            cls = self._TASK_TYPES.get(name)
            if cls is None:
                self.discover_task_types()
                cls = self._TASK_TYPES.get(name)
            if cls is None:
                raise ReasoningTypeError(f"Unknown reasoning type in combined expression: {name}")
            components.append(cls())
            component_names.append(name)

        # Create combined class dynamically
        class CombinedReasoning(BaseReasoning):
            def __init__(self, comps: List[BaseReasoning], comp_names: List[str]):
                super().__init__()
                self.components = comps
                self.component_names = comp_names
                self.name = "+".join(comp_names)

            def perform_reasoning(self, *args, **kwargs) -> Dict[str, Any]:
                # Extract initial context (default to empty dict)
                context = dict(kwargs.pop("context", {}) or {})
                # Keep original input for reference
                context["original_input"] = args if len(args) > 1 else (args[0] if args else None)

                step_results: Dict[str, Any] = {}
                for idx, (component, cname) in enumerate(zip(self.components, self.component_names), start=1):
                    # Pass context enriched with previous outputs
                    step_context = context.copy()
                    if idx > 1:
                        prev_key = f"step_{idx-1}_{self.component_names[idx-2]}"
                        step_context["prev_step_result"] = step_results.get(prev_key)
                        step_context[f"prev_{self.component_names[idx-2]}_result"] = step_results.get(prev_key)

                    step_result = component.perform_reasoning(*args, context=step_context, **kwargs)
                    step_results[f"step_{idx}_{cname}"] = step_result
                    step_results["prev_step_result"] = step_result

                    if isinstance(step_result, dict):
                        for key in ("best_explanation", "final_output", "result"):
                            if key in step_result:
                                step_results[key] = step_result[key]

                    context[f"step_{idx}_result"] = step_result
                    context["prev_step_result"] = step_result

                last_result = step_results.get("prev_step_result", {})
                return {
                    "combined_result": step_results,
                    "reasoning_types": self.name,
                    "final_output": last_result,
                }

        return CombinedReasoning(components, component_names)
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            cache_hits = 0
            cache_misses = 0
            if self._instance_cache is not None:
                metrics = self._instance_cache.metrics()
                cache_hits = metrics.get("hits", 0)
                cache_misses = metrics.get("misses", 0)
            return {
                "total_single_creations": self._stats.get("single", 0),
                "total_combined_creations": self._stats.get("combined", 0),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "registered_types": list(self._TASK_TYPES.keys()),
            }
    
    def shutdown(self, checkpoint_memory: bool = True) -> None:
        if checkpoint_memory:
            self.reasoning_memory.save_checkpoint()
        if self._instance_cache:
            self._instance_cache.clear()
        logger.info("ReasoningTypes shut down")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        """Convert value to float or None if value is None/empty string."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                "reasoning_types value must be numeric or None",
                cause=exc, context={"value": value}
            ) from exc

    # Backward‑compatible private method (used by legacy code)
    def _determine_reasoning_strategy(self, problem: str) -> str:
        return self.determine_reasoning_strategy(problem)


# ----------------------------------------------------------------------
# Self‑test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Reasoning Types ===\n")
    printer.status("TEST", "Reasoning Types initialized", "info")

    factory = ReasoningTypes()

    # 1. Single strategy creation
    abduction = factory.create("abduction")
    assert abduction is not None
    printer.status("PASS", "Created single abduction instance", "success")

    # 2. Combined strategy creation
    combined = factory.create("abduction+induction")
    assert combined is not None
    assert "combined_result" in combined.perform_reasoning(observations=["test"]) # type: ignore
    printer.status("PASS", "Created combined reasoning (3 types)", "success")

    # 3. Combined with invalid count
    try:
        factory.create("abduction+induction+deduction+cause_effect")
        assert False, "Should reject > max_combined_types"
    except ReasoningTypeError:
        printer.status("PASS", "Rejected too many combined types", "success")

    # 4. Strategy suggestion
    prob1 = "Explain why the grass is wet"
    assert factory.determine_reasoning_strategy(prob1) == "abduction"
    prob2 = "Prove that Socrates is mortal given premises"
    assert factory.determine_reasoning_strategy(prob2) == "deduction"
    prob3 = "Find patterns in temperature data"
    assert "induction" in factory.determine_reasoning_strategy(prob3)
    printer.status("PASS", "Keyword‑based strategy suggestion works", "success")

    # 5. Instance caching (if enabled)
    if factory.enable_instance_cache:
        first = factory.create("deduction")
        second = factory.create("deduction")
        assert first is second, "Cached instance should be reused"
        printer.status("PASS", "Instance cache working", "success")
    else:
        printer.status("SKIP", "Instance cache disabled", "warning")

    # 6. Memory integration
    memory = factory.get_memory()
    memory.add({"type": "test", "content": "reasoning_types_test"}, priority=0.8)
    assert memory.size() >= 1
    printer.status("PASS", "Memory integration OK", "success")

    # 7. Combined reasoning context passing
    class DummyReasoning(BaseReasoning):
        def perform_reasoning(self, *args, **kwargs):
            ctx = kwargs.get("context", {})
            prev = ctx.get("prev_step_result", None)
            return {"prev": prev, "step_info": "done"}

    # Register dummy for test (simulate component)
    factory._TASK_TYPES["dummy"] = DummyReasoning
    combo = factory.create("dummy+dummy")
    res = combo.perform_reasoning(input_data=42)
    assert "combined_result" in res
    # Check that second component received prev_step_result from first
    steps = res["combined_result"]
    assert steps["step_2_dummy"]["prev"] == steps["step_1_dummy"]
    printer.status("PASS", "Sequential context passing verified", "success")

    print("\n=== Test ran successfully ===\n")

