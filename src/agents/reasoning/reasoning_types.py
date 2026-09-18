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

from typing import Any, Dict, List, Mapping, Optional, Type, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from .types import *
from .reasoning_memory import ReasoningMemory
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

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        default_strategy: str = "deduction",
        memory: Optional[ReasoningMemory] = None,
    ) -> None:
        self.config: Dict[str, Any] = dict(config or load_global_config())
        self.types_cfg: Dict[str, Any] = dict(get_config_section("reasoning_types", self.config, default={}) or {})
        self.max_combined_types = bounded_iterations(self.types_cfg.get("max_combined_types", 3), minimum=1, maximum=10)
        # Phase 0 established that strategies contain mutable run state.
        # Therefore strategy instances remain request/session scoped.
        self.enable_instance_cache = False
        self._instance_cache = None
        self.default_strategy = (str(default_strategy).strip() or "deduction")
        self.strategy_keywords = dict(
            self.types_cfg.get(
                "strategy_keywords",
                {
                    "abduction": ["explain", "why", "hypothesis", "plausible", "most likely"],
                    "deduction": ["prove", "derive", "therefore", "premise", "must be"],
                    "induction": ["pattern", "trend", "generalize", "predict", "extrapolate"],
                    "analogical": ["analogy", "similar", "compare", "resembles"],
                    "decompositional": ["break down", "component", "decompose", "subsystem"],
                    "cause_effect": ["effect", "impact", "results in", "causal", "cause"],
                },
            )
        )

        # Borrowed dependency; ReasoningTypes does not own its lifetime.
        self.reasoning_memory = memory
        self._lock = threading.RLock()
        self._stats: Dict[str, int] = {"single": 0, "combined": 0}

        logger.info(
            "ReasoningTypes initialized | "
            "mutable_instance_cache=False | "
            "max_combined=%s | default=%s",
            self.max_combined_types,
            self.default_strategy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create(self, task_type: str) -> BaseReasoning:
        """Create a fresh reasoning strategy instance."""
        normalized = str(task_type or "").strip().lower()

        if not normalized:
            raise ReasoningTypeError("reasoning type is required", context={"task_type": task_type},)

        if "+" in normalized:
            instance = self._create_combined_reasoning(normalized)

            with self._lock:
                self._stats["combined"] += 1

            return instance

        instance = self._create_single_reasoning(normalized)

        with self._lock:
            self._stats["single"] += 1

        return instance

    def execute(self, task_type: str, problem: Any, context: Optional[Mapping[str, Any]] = None) -> Any:
        """Execute one or more reasoning strategies through one stable contract.

        ``problem`` is the top-level Agent input. ``context`` may provide the
        strategy-specific structured fields required by a concrete reasoner.
        """
        normalized = str(task_type or "").strip().lower()

        if not normalized:
            raise ReasoningTypeError(
                "reasoning type is required",
                context={
                    "task_type": task_type,
                },
            )

        context_dict = dict(context or {})

        if "+" in normalized:
            names = self._parse_combined_types(normalized)
            return self._execute_combined(names, problem, context_dict)

        component = self._create_single_reasoning(normalized)
        return self._invoke_component(normalized, component, problem, context_dict)


    def invoke_instance(
        self,
        reasoning_engine: BaseReasoning,
        problem: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Invoke an already-created strategy without signature reflection.

        Kept primarily for compatibility with existing ReasoningAgent internals
        and external callers that still create strategies explicitly.
        """
        context_dict = dict(context or {})

        component_names = getattr(
            reasoning_engine,
            "component_names",
            None,
        )

        if component_names:
            return self._execute_combined(
                list(component_names),
                problem,
                context_dict,
            )

        for strategy_name, strategy_cls in self._TASK_TYPES.items():
            if isinstance(
                reasoning_engine,
                strategy_cls,
            ):
                return self._invoke_component(
                    strategy_name,
                    reasoning_engine,
                    problem,
                    context_dict,
                )

        # Registered extension strategies are required to respect BaseReasoning's
        # declared input_data/context contract.
        performer = getattr(
            reasoning_engine,
            "perform_reasoning",
            None,
        )

        if not callable(performer):
            raise ReasoningTypeError(
                "Reasoning engine does not expose perform_reasoning",
                context={
                    "engine":
                        type(reasoning_engine).__name__,
                },
            )

        return performer(
            problem,
            context_dict,
        )

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

    def get_memory(self) -> Optional[ReasoningMemory]:
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
        """Create a compatibility wrapper for combined strategies."""
        names = self._parse_combined_types(
            combined_type
        )

        factory = self

        class CombinedReasoning(BaseReasoning):
            def __init__(self, component_names: List[str]) -> None:
                super().__init__()

                self.component_names = list(component_names)
                self.name = "+".join(self.component_names)

            def perform_reasoning(
                self,
                input_data: Any = None,
                context: Optional[Dict[str, Any]] = None,
                **legacy_kwargs: Any,
            ) -> Dict[str, Any]:
                merged_context = dict(context or {})

                for key, value in legacy_kwargs.items():
                    merged_context.setdefault(key, value)

                problem = input_data

                if problem is None:
                    for key in (
                        "problem",
                        "observations",
                        "events",
                        "system",
                        "target",
                        "hypothesis",
                    ):
                        if key in merged_context:
                            problem = merged_context[key]
                            break

                return factory._execute_combined(self.component_names, problem, merged_context)
        return CombinedReasoning(names)
    
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
        """
        Release ReasoningTypes-owned resources.

        ``checkpoint_memory`` remains in the signature for v2.3 caller
        compatibility but ReasoningTypes no longer owns memory persistence.
        """
        if checkpoint_memory:
            logger.debug("Ignoring checkpoint_memory=True: Reasoning memory durability is owned by BaseAgent/checkpointing.")

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

    def _parse_combined_types(
        self,
        combined_type: str,
    ) -> List[str]:
        names = [
            name.strip().lower()
            for name in combined_type.split("+")
            if name.strip()
        ]

        if not (
            1
            <= len(names)
            <= self.max_combined_types
        ):
            raise ReasoningTypeError(
                (
                    "Combined reasoning must include between "
                    f"1 and {self.max_combined_types} strategies"
                ),
                context={
                    "requested": len(names),
                    "max": self.max_combined_types,
                },
            )

        unknown = [
            name
            for name in names
            if name not in self._TASK_TYPES
        ]

        if unknown:
            self.discover_task_types()

            unknown = [
                name
                for name in names
                if name not in self._TASK_TYPES
            ]

        if unknown:
            raise ReasoningTypeError(
                "Unknown reasoning type(s) in combined expression",
                context={
                    "unknown": unknown,
                    "available": sorted(
                        self._TASK_TYPES.keys()
                    ),
                },
            )

        return names


    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return list(value)

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, set):
            return list(value)

        return [value]


    def _invoke_component(
        self,
        strategy_name: str,
        component: BaseReasoning,
        problem: Any,
        context: Mapping[str, Any],
    ) -> Any:
        """Adapt the canonical Reasoning request to one concrete strategy."""
        strategy = str(
            strategy_name
        ).strip().lower()

        context_dict = dict(context or {})

        problem_payload: Dict[str, Any] = (
            dict(problem)
            if isinstance(problem, Mapping)
            else {}
        )

        if strategy == "abduction":
            observations = context_dict.get(
                "observations",
                problem_payload.get(
                    "observations",
                    problem,
                ),
            )

            return getattr(component, "perform_reasoning")(  # type: ignore[call-arg]
                observations=observations,
                context=context_dict,
            )

        if strategy == "induction":
            observations = context_dict.get(
                "observations",
                problem_payload.get(
                    "observations",
                    problem,
                ),
            )

            return getattr(component, "perform_reasoning")(  # type: ignore[call-arg]
                observations=self._as_list(
                    observations
                ),
                context=context_dict,
            )

        if strategy == "deduction":
            premises = context_dict.get(
                "premises",
                problem_payload.get(
                    "premises",
                ),
            )

            hypothesis = context_dict.get(
                "hypothesis",
                problem_payload.get(
                    "hypothesis",
                ),
            )

            if premises is None:
                if isinstance(
                    problem,
                    (list, tuple, set),
                ):
                    premises = list(problem)
                else:
                    premises = [str(problem)]

            premises = [
                str(item)
                for item in self._as_list(
                    premises
                )
            ]

            if hypothesis is None:
                hypothesis = (
                    problem_payload.get("goal")
                    or problem_payload.get("query")
                    or str(problem)
                )

            hypothesis = str(hypothesis).strip()

            if not hypothesis:
                raise ReasoningValidationError(
                    "Deductive reasoning requires a hypothesis.",
                    context={
                        "strategy": strategy,
                    },
                )

            return getattr(component, "perform_reasoning")(  # type: ignore[call-arg]
                premises=premises,
                hypothesis=hypothesis,
                context=context_dict,
            )

        if strategy == "analogical":
            target = context_dict.get(
                "target",
                problem_payload.get(
                    "target",
                    problem,
                ),
            )

            source_domain = context_dict.get(
                "source_domain",
                problem_payload.get(
                    "source_domain",
                    [],
                ),
            )

            return getattr(component, "perform_reasoning")(  # type: ignore[call-arg]
                target=target,
                source_domain=self._as_list(
                    source_domain
                ),
                context=context_dict,
            )

        if strategy == "cause_effect":
            events = context_dict.get(
                "events",
                problem_payload.get(
                    "events",
                    problem,
                ),
            )

            conditions = context_dict.get(
                "conditions",
                problem_payload.get(
                    "conditions",
                    {},
                ),
            )

            return getattr(component, "perform_reasoning")(  # type: ignore[call-arg]
                events=self._as_list(events),
                conditions=dict(
                    conditions or {}
                ),
                context=context_dict,
            )

        if strategy == "decompositional":
            system = context_dict.get(
                "system",
                problem_payload.get(
                    "system",
                    problem,
                ),
            )

            return getattr(component, "perform_reasoning")(  # type: ignore[call-arg]
                system=system,
                context=context_dict,
            )

        # Extensions registered outside the built-in strategy set must implement
        # BaseReasoning's canonical input_data/context interface.
        return component.perform_reasoning(
            problem,
            context_dict,
        )


    def _execute_combined(
        self,
        names: List[str],
        problem: Any,
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Execute combined strategies with fresh per-step instances."""
        working_context = dict(context or {})

        working_context.setdefault(
            "original_input",
            problem,
        )

        step_results: Dict[str, Any] = {}
        previous_result: Any = None

        for index, strategy_name in enumerate(
            names,
            start=1,
        ):
            component = self._create_single_reasoning(
                strategy_name
            )

            step_context = dict(
                working_context
            )

            if previous_result is not None:
                step_context[
                    "prev_step_result"
                ] = previous_result

                previous_name = names[
                    index - 2
                ]

                step_context[
                    f"prev_{previous_name}_result"
                ] = previous_result

            result = self._invoke_component(
                strategy_name,
                component,
                problem,
                step_context,
            )

            step_key = (
                f"step_{index}_{strategy_name}"
            )

            step_results[
                step_key
            ] = result

            working_context[
                f"step_{index}_result"
            ] = result

            working_context[
                "prev_step_result"
            ] = result

            previous_result = result

        return {
            "combined_result": step_results,
            "reasoning_types": "+".join(names),
            "final_output": previous_result,
        }


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

