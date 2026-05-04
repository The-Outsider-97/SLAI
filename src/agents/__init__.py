"""Top-level exports for agent modules.

Uses lazy loading to avoid importing optional heavy dependencies at package import time.
"""

__version__ = "2.1.0"

_EXPORTS = {
    "AgentFactory": (".agent_factory", "AgentFactory"),
    "BaseAgent": (".base_agent", "BaseAgent"),
    "ExecutionRecord": (".base_agent", "ExecutionRecord"),
    "ResourceMonitor": (".base_agent", "ResourceMonitor"),
    "RetrainingManager": (".base_agent", "RetrainingManager"),
    "_ensure_torch_imported": (".base_agent", "_ensure_torch_imported"),
    "AdaptiveAgent": (".adaptive_agent", "AdaptiveAgent"),
    "AlignmentAgent": (".alignment_agent", "AlignmentAgent"),
    "BrowserAgent": (".browser_agent", "BrowserAgent"),
    "CollaborativeAgent": (".collaborative_agent", "CollaborativeAgent"),
    "EvaluationAgent": (".", "EvaluationAgent"),
    "ExecutionAgent": (".execution_agent", "ExecutionAgent"),
    "HandlerAgent": (".handler_agent", "HandlerAgent"), # done for 2.1.0 needs to be updated and expanded for 2.2.0
    "KnowledgeAgent": (".knowledge_agent", "KnowledgeAgent"),
    "LanguageAgent": (".language_agent", "LanguageAgent"),
    "LearningAgent": (".learning_agent", "LearningAgent"),
    "NetworkAgent": (".network_agent", "NetworkAgent"),
    "ObservabilityAgent": (".observability_agent", "ObservabilityAgent"),
    "PerceptionAgent": (".perception_agent", "PerceptionAgent"), # done for 2.1.0. Maybe split into multiple agents for 2.2.0 (e.g., vision agent, audio agent, encoder agent, etc.)
    "PlanningAgent": (".planning_agent", "PlanningAgent"),
    "PrivacyAgent": (".privacy_agent", "PrivacyAgent"),
    "QNNAgent": (".qnn_agent", "QNNAgent"),
    "QualityAgent": (".quality_agent", "QualityAgent"),
    "ReaderAgent": (".reader_agent", "ReaderAgent"), # done for 2.1.0 needs to be updated and expanded for 2.2.0
    "ReasoningAgent": (".reasoning_agent", "ReasoningAgent"),
    "SafetyAgent": (".safety_agent", "SafetyAgent"),
    # For later: add more agents like social agent, emotional agent, gamer agent, simulation agent, etc.
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = __import__(f"{__name__}{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

#from .adaptive_agent import AdaptiveAgent, EpisodeSummary # done for 2.1.0
#from .agent_factory import AgentFactory
#from .alignment_agent import AlignmentAgent, PolicyFeedback, CorrectionDecision, RiskAssessment, PolicyAdapter # done for 2.1.0
#from .base_agent import _ensure_torch_imported, ExecutionRecord, BaseAgent, RetrainingManager, ResourceMonitor # done for 2.1.0
#from .browser_agent import BrowserAgent, BrowserAgentOptions, BrowserAgentExecution # done for 2.1.0
#from .collaborative_agent import * # done for 2.1.0
#from .evaluation_agent import EvaluationAgent, FallbackEvaluatorAgent, AIValidationSuite # done for 2.1.0
#from .execution_agent import ExecutionAgent # done for 2.1.0
#from .handler_agent import HandlerAgent # done for 2.1.0 needs to be updated and expanded for 2.2.0
#from .knowledge_agent import KnowledgeAgent, cosine_sim # done for 2.1.0
#from .language_agent import * # done for 2.1.0
#from .learning_agent import LearningAgent # done for 2.1.0
#from .network_agent import NetworkAgent # done for 2.1.0
#from .observability_agent import ObservabilityAgent, _LEVEL_RANK # done for 2.1.0
#from .perception_agent import PerceptionAgent # done for 2.1.0. Maybe split into multiple agents for 2.2.0 (e.g., vision agent, audio agent, encoder agent, etc.)
#from .planning_agent import PlanningAgent, HTNPlanner, PartialOrderPlanner, AStarPlanner, ExplanatoryPlanner # done for 2.1.0
#from .privacy_agent import PrivacyAgent, PrivacyExecutionReport # done for 2.1.0
#from .qnn_agent import QNNAgent, QuantumGate, QuantumCircuitLayer, RNNMetaLearner, MetaLearner, Task, PerformanceEvaluator
#from .quality_agent import QualityAgent, SubsystemExecution, QualityAgentDecision # done for 2.1.0
#from .reader_agent import ReaderAgent # done for 2.1.0 needs to be updated and expanded for 2.2.0
#from .reasoning_agent import ReasoningAgent, identity_rule, transitive_rule # done for 2.1.0
#from .safety_agent import SafetyAgent, INCIDENT_RESPONSE

# __all__ = [
    # Adaptive
#    "AdaptiveAgent",
#    "EpisodeSummary",
#    # Agent Factory
#    "AgentFactory",
#    # Alignment
#    "AlignmentAgent",
#    "PolicyFeedback",
#    "CorrectionDecision",
#    "RiskAssessment",
#    "PolicyAdapter",
#    # Base
#    "_ensure_torch_imported",
#    "ExecutionRecord",
#    "BaseAgent",
#    "RetrainingManager",
#    "ResourceMonitor",
#    # Browser
#    "BrowserAgent",
#    "BrowserAgentOptions",
#    "BrowserAgentExecution",
#    # Collaborative
#    "RiskLevel",
#    "CollaborativeAgentMode",
#    "CollaborativeAgentEventType",
#    "SafetyAssessment",
#    "CoordinationAssignment",
 #   "CoordinationResult",
 #   "DelegationRecord",
 #   "CollaborativeAgentConfig",
 #   "BayesianRiskModel",
#    "CollaborativeAgent",
#    # Evaluation
#    "EvaluationAgent",
#    "FallbackEvaluatorAgent",
#    "AIValidationSuite",
#    # Execution
#    "ExecutionAgent",
#    # Handler
#    "HandlerAgent",
#    # Knowledge
#    "KnowledgeAgent",
#    "cosine_sim",
#    # Language
#    "LanguageAgent",
#    "LanguageAgentResponse",
#    "PipelineTrace",
#    "PipelineArtifacts",
#    "StageRecord",
#    "PipelineStatus",
#    "StageName",
#    "StagePolicy",
#    "LanguageAgentRuntimeError",
#    "LanguageAgentConfigurationError",
#    # Learning
#    "LearningAgent",
#    # Network
#    "NetworkAgent",
#    # Observability
#    "ObservabilityAgent",
#    "_LEVEL_RANK",
#    # Perception
#    "PerceptionAgent",
#    # Planning
#    "PlanningAgent",
#    "HTNPlanner",
#    "PartialOrderPlanner",
#    "AStarPlanner",
#    "ExplanatoryPlanner",
#    # Privacy
#    "PrivacyAgent",
#    "PrivacyExecutionReport",
#    # QNN
#    "QNNAgent",
#    "QuantumGate",
#    "QuantumCircuitLayer",
#    "RNNMetaLearner",
#    "MetaLearner",
#    "Task",
#    "PerformanceEvaluator",
    # Quality
#    "QualityAgent",
#    "SubsystemExecution",
#    "QualityAgentDecision",
#    # Reader
#    "ReaderAgent",
#    # Reasoning
#    "ReasoningAgent",
#    "identity_rule",
#    "transitive_rule",
#    # Safety
#    "SafetyAgent",
#    "INCIDENT_RESPONSE"
# ]
