
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
    """
    Selects the best reasoning type for the task or combines up to 3 types for more copmplex ones
    """
    _task_types: Dict[str, Type[BaseReasoning]] = {
        'abduction': ReasoningAbduction,
        'deduction': ReasoningDeductive,
        'induction': ReasoningInductive,
        'analitical': ReasoningAnalogical,
        'decompositional': ReasoningDecompositional,
        'cause_effect': ReasoningCauseAndEffect,
    }

    def __init__(self):
        self.config = load_global_config()
        self.types_config = get_config_section("reasoning_types")
        self.memory = ReasoningMemory()

    def discover_task_types(self):
        import src.agents.reasoning.types  # root module
        for name, obj in inspect.getmembers(src.agents.reasoning.types):
            if inspect.isclass(obj) and issubclass(obj, BaseReasoning):
                self._task_types[obj.__name__.lower()] = obj

    def create(self, task_type: str) -> BaseReasoning:
        """Creates an instance of a specified reasoning type"""
        printer.status("CREATE", f"Request to create type of reasoning: '{task_type}'")
        
        # Normalize task type name
        normalized_type = task_type.lower().strip()
        
        # Check if it's a combined type (e.g., "abduction+induction")
        if '+' in normalized_type:
            return self._create_combined_reasoning(normalized_type)
        
        # Get the reasoning class
        reasoning_class = self._get_reasoning_class(normalized_type)
        if not reasoning_class:
            raise ValueError(f"Unknown reasoning type: {task_type}")
            
        # Instantiate the reasoning class
        instance = reasoning_class()
        printer.status("CREATED", f"Instance of {reasoning_class.__name__} created", "success")
        return instance

    def _get_reasoning_class(self, task_type: str) -> Optional[Type[BaseReasoning]]:
        """Get reasoning class by type name"""
        # Check predefined types
        if task_type in self._task_types:
            return self._task_types[task_type]
            
        # Check discovered types
        self.discover_task_types()
        return self._task_types.get(task_type)

    def _create_combined_reasoning(self, combined_type: str) -> BaseReasoning:
        """Create a combined reasoning instance with up to 3 types"""
        type_names = combined_type.split('+')
        if len(type_names) > 3:
            raise ValueError("Cannot combine more than 3 reasoning types")
            
        # Get individual reasoning classes
        reasoning_classes = []
        for t in type_names:
            cls = self._get_reasoning_class(t.strip())
            if not cls:
                raise ValueError(f"Unknown reasoning type in combination: {t}")
            reasoning_classes.append(cls)
            
        # Create combined reasoning class
        combined_class = self._create_combined_class(reasoning_classes)
        instance = combined_class()
        printer.status("CREATED", 
                      f"Combined reasoning instance: {combined_type}", 
                      "success")
        return instance

    def _create_combined_class(self, 
                              reasoning_classes: List[Type[BaseReasoning]]
                              ) -> Type[BaseReasoning]:
        """Dynamically create a combined reasoning class"""
        class CombinedReasoning(BaseReasoning):
            def __init__(self):
                super().__init__()
                self.components = [cls() for cls in reasoning_classes]
                self.name = "+".join([cls.__name__ for cls in reasoning_classes])
                
            def perform_reasoning(self, *args, **kwargs) -> Dict[str, Any]:
                """Execute reasoning components in sequence"""
                results = {}
                context = kwargs.get("context", {})
                
                for i, component in enumerate(self.components):
                    # Execute component with accumulated context
                    result = component.perform_reasoning(*args, **kwargs, context=context)
                    results[f"step_{i+1}_{type(component).__name__}"] = result
                    
                    # Update context for next component
                    context.update({
                        f"prev_{type(component).__name__}_result": result,
                        f"prev_step_result": result
                    })
                    
                # Create final combined result
                return {
                    "combined_result": results,
                    "reasoning_types": self.name,
                    "final_output": self._synthesize_result(results)
                }
                
            def _synthesize_result(self, results: Dict) -> Dict:
                """Synthesize final result from component outputs"""
                # Default implementation - override for custom synthesis
                last_key = list(results.keys())[-1]
                return results[last_key]
                
        # Set a meaningful name for the class
        class_names = [cls.__name__ for cls in reasoning_classes]
        CombinedReasoning.__name__ = "Combined_" + "_".join(class_names)
        return CombinedReasoning

    def _determine_reasoning_strategy(self, problem: str) -> str:
        """
        Dynamically select reasoning strategy based on problem analysis
        Returns either a single type or combined types (e.g., "abduction+deduction")
        """
        problem_lower = problem.lower()
    
        if any(word in problem_lower for word in [
            "explain", "why", "hypothesis", "plausible", "assumption", "guess", "possible cause", 
            "most likely", "reason for", "root cause"
        ]):
            return "abduction"
    
        if any(word in problem_lower for word in [
            "prove", "derive", "therefore", "implies", "logical consequence", "from this", 
            "deduce", "conclude", "premise", "certainty", "must be"
        ]):
            return "deduction"
    
        if any(word in problem_lower for word in [
            "generalize", "pattern", "trend", "predict", "examples suggest", "correlation", 
            "extrapolate", "inductive", "likely outcome", "probable result"
        ]):
            return "induction"
    
        if any(word in problem_lower for word in [
            "similar to", "analogy", "is like", "map between", "analogous", "compare with", 
            "corresponds to", "metaphor", "analogy-based", "resembles"
        ]):
            return "analitical"
    
        if any(word in problem_lower for word in [
            "break down", "decompose", "component", "subsystem", "analyze parts", 
            "structure", "function", "hierarchy", "system analysis", "modular"
        ]):
            return "decompositional"
    
        if any(word in problem_lower for word in [
            "cause", "effect", "leads to", "results in", "trigger", "because of", 
            "impact", "influence", "chain reaction", "causal", "outcome"
        ]):
            return "cause_effect"
        
        # Default combination for complex problems
        return "abduction+deduction"