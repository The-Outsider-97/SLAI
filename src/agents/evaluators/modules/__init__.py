"""Evaluator module exports loaded on demand.

EvaluationTransformer uses optional PyTorch. Importing the calculations,
static-analysis, or validation protocol packages must not import torch.
Legacy named imports and __all__ remain available through PEP 562.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "EvaluationTransformerOutput": "evaluation_transformer",
    "SinusoidalEvaluationPositionalEncoding": "evaluation_transformer",
    "EvaluationTransformer": "evaluation_transformer",
    "PerformanceMetrics": "evaluators_calculations",
    "EvaluatorsCalculations": "evaluators_calculations",
    "AnalysisIssue": "static_analyzer",
    "ParseResult": "static_analyzer",
    "FileAnalysisResult": "static_analyzer",
    "StaticAnalysisReport": "static_analyzer",
    "StaticAnalyzer": "static_analyzer",
    "ASTAnalyzer": "static_analyzer",
    "SymbolicExecutor": "static_analyzer",
    "DataFlowAnalyzer": "static_analyzer",
    "DataFlowVisitor": "static_analyzer",
    "ValidationProtocol": "validation_protocol",
    "ValidationIssue": "validation_protocol",
    "ValidationProtocolReport": "validation_protocol",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
