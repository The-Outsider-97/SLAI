"""Structured, deterministic financial reporting for SLAI v2.4.

Report owns presentation-neutral semantic sections and safe JSON/Markdown
rendering only, never PDF/DOCX/layout engines, calculations, model inference,
or external actions. Source/assumption evidence is preserved from FinancialResult.
Financial results may contain sensitive information: authorization, masking,
retention and storage are the caller's responsibility. Log only report metadata.
"""
from __future__ import annotations

__version__ = "2.4.0"

import html
import json
import re

from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any

from .financial_types import FinancialResult
from .utils.config_loader import load_global_config, get_config_section
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Report")
printer = PrettyPrinter()


def _escape_markdown(value: str) -> str:
    """Render untrusted labels as literal text, not markdown/HTML syntax."""
    escaped = html.escape(value, quote=True)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", escaped).replace("\n", " ").replace("\r", " ")


def _fenced_json(value: Any) -> str:
    payload = json.dumps(to_json_compatible(value), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
    # A long enough fence prevents user-supplied backticks from closing the block.
    runs = [len(token) for token in re.findall(r"`+", payload)]
    fence = "`" * max(3, max(runs, default=0) + 1)
    return f"{fence}json\n{payload}\n{fence}"


@dataclass(frozen=True, slots=True)
class FinancialReportSection:
    title: str
    content: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", require_text(self.title, "section.title", max_length=128))
        object.__setattr__(self, "content", freeze_mapping(self.content, "section.content"))

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "content": to_json_compatible(self.content)}


@dataclass(frozen=True, slots=True)
class FinancialReportDocument:
    title: str
    request_id: str
    operation: str
    status: str
    generated_at: datetime
    sections: tuple[FinancialReportSection, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", require_text(self.title, "report.title", max_length=256))
        object.__setattr__(self, "request_id", require_text(self.request_id, "request_id", max_length=128))
        object.__setattr__(self, "operation", require_text(self.operation, "operation", max_length=128))
        object.__setattr__(self, "status", require_text(self.status, "status", max_length=64))
        object.__setattr__(self, "generated_at", to_utc_datetime(self.generated_at, "generated_at"))
        if not isinstance(self.sections, (tuple, list)) or not all(isinstance(s, FinancialReportSection) for s in self.sections):
            raise FinancialValidationError("Report sections must contain FinancialReportSection values")
        object.__setattr__(self, "sections", tuple(self.sections))
        if not isinstance(self.warnings, (list, tuple)):
            raise FinancialValidationError("Report warnings must be list/tuple")
        object.__setattr__(self, "warnings", tuple(require_text(w, "warning", max_length=1024) for w in self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1", "title": self.title, "request_id": self.request_id,
                "operation": self.operation, "status": self.status,
                "generated_at": iso_utc(self.generated_at),
                "warnings": list(self.warnings),
                "sections": [section.to_dict() for section in self.sections]}


class FinancialReport:
    """Finance-neutral report assembly from validated FinancialResult evidence."""

    def __init__(self):
        self.config = load_global_config()
        self.financial_report_config = get_config_section("financial_report", config=self.config)
        try:
            self.max_sections = require_positive_int(self.financial_report_config["max_sections"], "max_sections", maximum=256)
            self.max_output_chars = require_positive_int(self.financial_report_config["max_output_chars"], "max_output_chars", maximum=50000000)
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_report configuration key", context={"key": str(exc)}) from exc

    def from_result(self, result: FinancialResult, *, title: str | None = None) -> FinancialReportDocument:
        if not isinstance(result, FinancialResult):
            raise FinancialValidationError("Report input must be FinancialResult")
        resolved = (require_text(title, "title", max_length=256) if title is not None
                    else f"Financial {result.domain}: {result.operation}")
        sections: list[FinancialReportSection] = []
        # Result is a single value object and may not be a mapping. Preserve it.
        sections.append(FinancialReportSection("Analysis", {
            "result": result.result, "status": result.status,
            "completed_at": iso_utc(result.completed_at)}))
        if result.assumptions:
            sections.append(FinancialReportSection("Assumptions", result.assumptions))
        if result.diagnostics:
            sections.append(FinancialReportSection("Diagnostics", result.diagnostics))
        if result.sources:
            sections.append(FinancialReportSection("Sources", {"items": [source.to_dict() for source in result.sources]}))
        if result.methodology or result.model_version:
            sections.append(FinancialReportSection("Methodology", {"description": result.methodology, "model_version": result.model_version}))
        if result.metadata:
            sections.append(FinancialReportSection("Metadata", result.metadata))
        if len(sections) > self.max_sections:
            raise FinancialSerializationError("Report requires more than configured max_sections")
        document = FinancialReportDocument(resolved, result.request_id, result.operation,
                                           result.status, datetime.now(timezone.utc), tuple(sections), result.warnings)
        logger.debug("Financial report assembled | operation=%s | sections=%s", result.operation, len(sections))
        return document

    def to_json(self, report: FinancialReportDocument, *, pretty: bool = True) -> str:
        if not isinstance(report, FinancialReportDocument):
            raise FinancialValidationError("report must be FinancialReportDocument")
        result = json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2 if pretty else None)
        if len(result) > self.max_output_chars:
            raise FinancialSerializationError("JSON report exceeds configured max_output_chars")
        return result

    def to_markdown(self, report: FinancialReportDocument) -> str:
        if not isinstance(report, FinancialReportDocument):
            raise FinancialValidationError("report must be FinancialReportDocument")
        lines = [f"# {_escape_markdown(report.title)}", "",
                 f"**Request:** {_escape_markdown(report.request_id)}",
                 f"**Operation:** {_escape_markdown(report.operation)}",
                 f"**Status:** {_escape_markdown(report.status)}",
                 f"**Generated (UTC):** {iso_utc(report.generated_at)}", ""]
        if report.warnings:
            lines.extend(["## Warnings", ""])
            lines.extend(f"- {_escape_markdown(warning)}" for warning in report.warnings)
            lines.append("")
        for section in report.sections:
            lines.extend([f"## {_escape_markdown(section.title)}", "", _fenced_json(section.content), ""])
        result = "\n".join(lines).rstrip() + "\n"
        if len(result) > self.max_output_chars:
            raise FinancialSerializationError("Markdown report exceeds configured max_output_chars")
        return result

    def to_html(self, report: FinancialReportDocument) -> str:
        """Render a simple HTML report for web display, not PDF or DOCX."""
        if not isinstance(report, FinancialReportDocument):
            raise FinancialValidationError("report must be FinancialReportDocument")
        lines = [f"<h1>{html.escape(report.title)}</h1>",
                 f"<p><strong>Request:</strong> {html.escape(report.request_id)}<br>",
                 f"<strong>Operation:</strong> {html.escape(report.operation)}<br>",
                 f"<strong>Status:</strong> {html.escape(report.status)}<br>",
                 f"<strong>Generated (UTC):</strong> {iso_utc(report.generated_at)}</p>"]
        if report.warnings:
            lines.append("<h2>Warnings</h2><ul>")
            lines.extend(f"<li>{html.escape(warning)}</li>" for warning in report.warnings)
            lines.append("</ul>")
        for section in report.sections:
            lines.append(f"<h2>{html.escape(section.title)}</h2>")
            lines.append(f"<pre>{html.escape(json.dumps(to_json_compatible(section.content), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))}</pre>")
        result = "\n".join(lines).rstrip() + "\n"
        if len(result) > self.max_output_chars:
            raise FinancialSerializationError("HTML report exceeds configured max_output_chars")
        return result

    def to_sldc(self, report: FinancialReportDocument) -> str:
        """Render a simple SLDC report for SLAI's internal display, not PDF or DOCX."""
        if not isinstance(report, FinancialReportDocument):
            raise FinancialValidationError("report must be FinancialReportDocument")
        lines = [f"SLDC REPORT | {report.title} | Request: {report.request_id} | Operation: {report.operation} | Status: {report.status} | Generated (UTC): {iso_utc(report.generated_at)}"]
        if report.warnings:
            lines.append("WARNINGS:")
            lines.extend(f"- {warning}" for warning in report.warnings)
        for section in report.sections:
            lines.append(f"SECTION: {section.title}")
            lines.append(json.dumps(to_json_compatible(section.content), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
        result = "\n".join(lines).rstrip() + "\n"
        if len(result) > self.max_output_chars:
            raise FinancialSerializationError("SLDC report exceeds configured max_output_chars")
        return result


__all__ = ["FinancialReportSection", "FinancialReportDocument", "FinancialReport"]

if __name__ == "__main__":
    print("\n=== Running Financial Report ===\n")
    printer.status("TEST", "Financial Report initialized", "info")
    from .financial_types import FinancialResult
    reporter = FinancialReport()
    result = FinancialResult("report-demo", "calculation", "npv", "success",
                             {"amount": "100.00", "currency": "EUR"},
                             assumptions={"discount_rate": "0.05"}, warnings=("Estimated only",))
    report = reporter.from_result(result, title="NPV Analysis")
    assert len(report.sections) == 2
    assert json.loads(reporter.to_json(report))["request_id"] == "report-demo"
    assert "Estimated only" in reporter.to_markdown(report)
    assert "## Assumptions" in reporter.to_markdown(report)
    print("\n=== Test ran successfully ===\n")
