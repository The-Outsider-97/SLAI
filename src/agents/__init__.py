"""Top-level exports for agent modules.

Uses lazy loading to avoid importing optional heavy dependencies at package import time.
"""

__version__ = "2.2.0"

_EXPORTS = {
    "AgentFactory": (".agent_factory", "AgentFactory"),
    "BaseAgent": (".base_agent", "BaseAgent"),
    "ExecutionRecord": (".base_agent", "ExecutionRecord"),
    "ResourceMonitor": (".base_agent", "ResourceMonitor"),
    "RetrainingManager": (".base_agent", "RetrainingManager"),
    "_ensure_torch_imported": (".base_agent", "_ensure_torch_imported"),
    "AgentRuntimeIdentity": (".runtime_contracts", "AgentRuntimeIdentity"),
    "AutonomousControlLoop": (".autonomous_control_loop", "AutonomousControlLoop"),
    "AutonomousLoopConfig": (".autonomous_control_loop", "AutonomousLoopConfig"),
    "ControlLoopState": (".autonomous_control_loop", "ControlLoopState"),
    "RuntimeHealth": (".runtime_contracts", "RuntimeHealth"),
    "RuntimeLifecycle": (".runtime_contracts", "RuntimeLifecycle"),
    "RuntimeStatus": (".runtime_contracts", "RuntimeStatus"),

    "AdaptiveAgent": (".adaptive_agent", "AdaptiveAgent"),
    "AlignmentAgent": (".alignment_agent", "AlignmentAgent"),
    "BrowserAgent": (".browser_agent", "BrowserAgent"),
    "CollaborativeAgent": (".collaborative_agent", "CollaborativeAgent"),
    "EvaluationAgent": (".evaluation_agent", "EvaluationAgent"),
    "ExecutionAgent": (".execution_agent", "ExecutionAgent"),
    "HandlerAgent": (".handler_agent", "HandlerAgent"), # done for 2.1.0 needs to be updated and expanded for 2.3.0
    "KnowledgeAgent": (".knowledge_agent", "KnowledgeAgent"),
    "LanguageAgent": (".language_agent", "LanguageAgent"),
    "LearningAgent": (".learning_agent", "LearningAgent"),
    "NetworkAgent": (".network_agent", "NetworkAgent"),
    "ObservabilityAgent": (".observability_agent", "ObservabilityAgent"),
    "PerceptionAgent": (".perception_agent", "PerceptionAgent"), # done for 2.1.0. Maybe split into multiple agents for 2.3.0 (e.g., vision agent, audio agent, encoder agent, etc.)
    "PlanningAgent": (".planning_agent", "PlanningAgent"),
    "PrivacyAgent": (".privacy_agent", "PrivacyAgent"),
    "QNNAgent": (".qnn_agent", "QNNAgent"),
    "QualityAgent": (".quality_agent", "QualityAgent"),
    "ReaderAgent": (".reader_agent", "ReaderAgent"), # done for 2.1.0 needs to be updated and expanded for 2.3.0
    "ReasoningAgent": (".reasoning_agent", "ReasoningAgent"),
    "SafetyAgent": (".safety_agent", "SafetyAgent"),
    # For later: add more agents like social agent, emotional agent, gamer agent, simulation agent, etc.
}

__all__ = sorted(_EXPORTS.keys()) # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = __import__(f"{__name__}{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
