"""Fail-closed per-record access policy. Authentication is external."""
from __future__ import annotations

from .contracts import AccessContext, AuthorizationDenied, RecordDraft, Scope
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.knowledge_errors import *
from ..utils.knowledge_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("assurance ACL")
printer = PrettyPrinter()


def can_read(draft: RecordDraft, ctx: AccessContext | None) -> bool:
    if draft.scope == Scope.GLOBAL:
        return True
    if ctx is None:
        return False
    if draft.scope == Scope.PROJECT:
        return bool(draft.project_id and draft.project_id in ctx.project_ids)
    if draft.scope == Scope.ORGANIZATION:
        return bool(draft.organization_id and draft.organization_id in ctx.organization_ids)
    return False


def require_read(draft: RecordDraft, ctx: AccessContext | None) -> None:
    if not can_read(draft, ctx):
        raise AuthorizationDenied("Knowledge record is not accessible in this context", action="read")


def require_write(draft: RecordDraft, ctx: AccessContext) -> None:
    if not ctx.can_write:
        raise AuthorizationDenied("Write permission required", action="write")
    if draft.scope == Scope.GLOBAL and not ctx.can_publish_global:
        raise AuthorizationDenied("Global knowledge publication permission required", action="publish_global")
    if draft.scope == Scope.PROJECT and draft.project_id not in ctx.project_ids:
        raise AuthorizationDenied("Project membership required", action="write")
    if draft.scope == Scope.ORGANIZATION and draft.organization_id not in ctx.organization_ids:
        raise AuthorizationDenied("Organization membership required", action="write")
    if draft.status.value in ("reviewed", "active") and not ctx.can_review:
        raise AuthorizationDenied("Approval permission required to activate/review knowledge", action="review")


__all__ = [
    "can_read",
    "require_read",
    "require_write",
]