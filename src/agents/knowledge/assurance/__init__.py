"""Opt-in, versioned Knowledge Assurance components for SLAI v2.4.

Do not confuse these proposed contracts with existing KnowledgeAgent APIs.
Legacy KnowledgeAgent remains separate until its integration gates are passed.
"""
from .contracts import *
from .repository import KnowledgeRepository
from .gateway import KnowledgeAssuranceGateway
from .project import ProjectKnowledgeService
from .standards import StandardsRegistry
from .rules import DeclarativeRuleEvaluator, RulesRegistry

__all__ = ["AccessContext", "RecordDraft", "RecordRevision", "Scope", "Lifecycle", "RuleOutcome",
           "RuleEvaluation", "Applicability", "KnowledgeRepository", "KnowledgeAssuranceGateway",
           "ProjectKnowledgeService", "StandardsRegistry", "DeclarativeRuleEvaluator", "RulesRegistry"]

from .repository import KnowledgeOutbox, OutboxDelivery, KnowledgeBackupService
from .rules import AssessmentService, AssessmentSnapshot
from .project import (ProjectAdoptionService, AdoptionResolution,
                      HistoricalLineageService, KnowledgePromotionService)
from .standards import PublisherUpdateReview
from .gateway import (EnhancedKnowledgeGateway, ScopedLexicalRetriever, RetrievalHit,
                      RetrievalResult, KnowledgeSufficiencyService, Sufficiency,
                      SufficiencyReport, OutboxDispatcher, DispatchReport, AssuranceDiagnostics,
                      OperationalHealth, RetrievalBenchmarkCase, RetrievalBenchmarkResult,
                      load_cases, run_benchmark)
__all__ += [
 "AssessmentService", "AssessmentSnapshot", "ProjectAdoptionService", "AdoptionResolution",
 "HistoricalLineageService", "KnowledgePromotionService", "KnowledgeOutbox", "OutboxDelivery",
 "EnhancedKnowledgeGateway", "ScopedLexicalRetriever", "RetrievalHit", "RetrievalResult",
 "KnowledgeSufficiencyService", "Sufficiency", "SufficiencyReport", "OutboxDispatcher",
 "DispatchReport", "AssuranceDiagnostics", "OperationalHealth", "KnowledgeBackupService",
 "PublisherUpdateReview", "RetrievalBenchmarkCase", "RetrievalBenchmarkResult", "load_cases",
 "run_benchmark",
]
