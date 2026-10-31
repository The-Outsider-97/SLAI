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
