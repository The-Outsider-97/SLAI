"""Verification entrypoints, structured validation reports and conversion fidelity diagnostics."""
from .invariants import validate_document
from .loss_report import ConversionReport
from .report import Issue, ValidationReport, check_drawing, check_scene, report_from_exception

__all__ = ["validate_document", "ConversionReport", "Issue", "ValidationReport", "check_drawing",
           "check_scene", "report_from_exception"]
