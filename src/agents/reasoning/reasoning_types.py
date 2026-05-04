
import inspect

from typing import Any, Dict, List, Optional, Type

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.types.base_reasoning import BaseReasoning
from src.agents.reasoning.types.reasoning_abduction import ReasoningAbduction
from src.agents.reasoning.types.reasoning_deductive import ReasoningDeductive
from src.agents.reasoning.types.reasoning_inductive import ReasoningInductive
from src.agents.reasoning.types.reasoning_analogical import ReasoningAnalogical
from src.agents.reasoning.types.reasoning_decompositional import ReasoningDecompositional
from src.agents.reasoning.types.reasoning_cause_effect import ReasoningCauseAndEffect
from src.agents.reasoning.reasoning_memory import ReasoningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Types")
printer = PrettyPrinter

class ReasoningTypes:
    """Factory/policy layer for single and combined reasoning strategies."""

    _task_types: Dict[str, Type[BaseReasoning]] = {
        "abduction": ReasoningAbduction,
        "deduction": ReasoningDeductive,
        "induction": ReasoningInductive,
        "analogical": ReasoningAnalogical,
        "analitical": ReasoningAnalogical,  # backward-compat typo
        "decompositional": ReasoningDecompositional,
        "cause_effect": ReasoningCauseAndEffect,
    }

    def discover_task_types(self) -> None:
        import src.agents.reasoning.types as reasoning_types_module

        for _, obj in inspect.getmembers(reasoning_types_module):
            if inspect.isclass(obj) and issubclass(obj, BaseReasoning) and obj is not BaseReasoning:
                normalized = obj.__name__.replace("Reasoning", "").lower()
                self._task_types.setdefault(normalized, obj)

    def create(self, task_type: str) -> BaseReasoning:
        normalized = (task_type or "").strip().lower()
        if not normalized:
            raise ValueError("reasoning type is required")

        if "+" in normalized:
            return self._create_combined_reasoning(normalized)

        reasoning_cls = self._task_types.get(normalized)
        if reasoning_cls is None:
            self.discover_task_types()
            reasoning_cls = self._task_types.get(normalized)

        if reasoning_cls is None:
            raise ValueError(f"Unknown reasoning type: {task_type}")
        return reasoning_cls()

    def _create_combined_reasoning(self, combined_type: str) -> BaseReasoning:
        names = [n.strip() for n in combined_type.split("+") if n.strip()]
        if not 1 <= len(names) <= 3:
            raise ValueError("Combined reasoning must include between 1 and 3 strategies")
        classes = [self._task_types.get(name) for name in names]
        if any(cls is None for cls in classes):
            self.discover_task_types()
            classes = [self._task_types.get(name) for name in names]
        if any(cls is None for cls in classes):
            missing = [name for name, cls in zip(names, classes) if cls is None]
            raise ValueError(f"Unknown reasoning type(s): {', '.join(missing)}")

        class CombinedReasoning(BaseReasoning):
            def __init__(self):
                super().__init__()
                self.components = [cls() for cls in classes]
                self.name = "+".join(type(c).__name__ for c in self.components)

            def perform_reasoning(self, *args, **kwargs) -> Dict[str, Any]:
                context = dict(kwargs.pop("context", {}) or {})
                output: Dict[str, Any] = {}
                for idx, component in enumerate(self.components, start=1):
                    result = component.perform_reasoning(*args, context=context, **kwargs)
                    output[f"step_{idx}_{type(component).__name__}"] = result
                    context[f"step_{idx}"] = result
                last = next(reversed(output.values())) if output else {}
                return {
                    "combined_result": output,
                    "reasoning_types": self.name,
                    "final_output": last,
                }

        return CombinedReasoning()

    def determine_reasoning_strategy(self, problem: str) -> str:
        text = (problem or "").lower()
        if any(w in text for w in ["explain", "why", "hypothesis", "plausible", "most likely"]):
            return "abduction"
        if any(w in text for w in ["prove", "derive", "therefore", "premise", "must be"]):
            return "deduction"
        if any(w in text for w in ["pattern", "trend", "generalize", "predict", "extrapolate"]):
            return "induction"
        if any(w in text for w in ["analogy", "similar", "compare", "resembles"]):
            return "analogical"
        if any(w in text for w in ["break down", "component", "decompose", "subsystem"]):
            return "decompositional"
        if any(w in text for w in ["effect", "impact", "results in", "causal", "cause"]):
            return "cause_effect"
        return "abduction+deduction"

    # backward-compatible private name used by existing code
    def _determine_reasoning_strategy(self, problem: str) -> str:
        return self.determine_reasoning_strategy(problem)
