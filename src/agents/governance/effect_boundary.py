"""Authorisation at the side-effect boundary, with evidence of what is actually covered (GOV-P0-03).

Planning-time authorisation is necessary but not sufficient: the legal question is whether *this effect* (a subprocess, an
outbound request, a file export, a dispatched command) is covered by an authorisation that is **still valid right now**.

Pieces
------
``governed_scope``      ambient (contextvar) scope opened by ``BaseAgent.execute`` once the gate has decided. Nested agent calls,
                        delegations and effects *inherit* it: they check the lease, they do not re-run the policy engine, so a
                        multi-agent workflow is evaluated once per distinct request, not once per hop.
``EffectBoundary``      ``guard(kind, target, ...)`` is called at the real choke points. Inside a scope it verifies the lease
                        (``AuthorizationRevoked`` when facts changed - the TOCTOU defence) and that the effect stays inside the
                        authorised destination/recipients. Outside a scope, low-risk internal kinds pass; governed kinds are
                        refused in ``enforce`` mode and reported as ``would_block`` in ``observe`` mode.
``GovernanceCommandAuthority``
                        composes with the existing ``CommandGateway`` authority (identity/permissions stay with it): the
                        gateway's own decision is a precondition, governance adds the regulatory check. No parallel gateway.
coverage reports        ``scan_effect_sites`` is *static* evidence (which call sites are wrapped); ``EffectBoundary.coverage`` is
                        *runtime* evidence (which boundaries actually fired, in which scope, with what outcome). They are never
                        merged into one number.

Stdlib only.
"""

from __future__ import annotations

import ast
import contextvars
import dataclasses

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .processing_context import Decision, GovernanceDenied, ProcessingContext, ProcessingDecision, RuleResult, new_id
from .reassessment import AuthorizationRevoked, ReassessmentEngine

__all__ = [
    "EFFECT_KINDS", "EffectBoundary", "EffectPermit", "GovernanceCommandAuthority", "GovernedScope", "current_scope",
    "governed_scope", "guard_effect", "install_effect_boundary", "installed_effect_boundary", "scan_effect_sites",
]

#: kind -> (governed?, description). Ungoverned kinds are internal, reversible and carry no external disclosure.
EFFECT_KINDS: Dict[str, Tuple[bool, str]] = {
    "compute_internal": (False, "in-process computation"),
    "read_internal": (False, "read of process-local state"),
    "subprocess": (True, "spawning a process"),
    "network_outbound": (True, "outbound network request"),
    "file_write": (True, "writing outside process memory"),
    "export": (True, "export / disclosure of data"),
    "command": (True, "dispatch of a gateway command"),
    "model_training": (True, "training or fine-tuning"),
    "tool": (True, "invoking an external tool or application"),
}


@dataclasses.dataclass
class GovernedScope:
    agent: str
    decision: ProcessingDecision
    ctx: Optional[ProcessingContext]
    auth_id: Optional[str]
    parent: Optional["GovernedScope"] = None
    depth: int = 0
    children: Dict[str, ProcessingDecision] = dataclasses.field(default_factory=dict)


_scope: "contextvars.ContextVar[Optional[GovernedScope]]" = contextvars.ContextVar("slai_governed_scope", default=None)


def current_scope() -> Optional[GovernedScope]:
    return _scope.get()


@contextmanager
def governed_scope(agent: str, decision: Optional[ProcessingDecision], ctx: Optional[ProcessingContext] = None) -> Iterator[Optional[GovernedScope]]:
    """Open the ambient scope for an executing agent. A no-op while no decision exists (gate not installed / mode off)."""
    if decision is None or not decision.allowed:
        # No decision (gate off) or a non-permitting one under observe mode: nothing was authorised, so no scope is opened.
        yield None
        return
    parent = _scope.get()
    scope = GovernedScope(agent, decision, ctx, decision.authorization_id, parent, (parent.depth + 1) if parent else 0)
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


@dataclasses.dataclass(frozen=True)
class EffectPermit:
    kind: str
    target: str
    auth_id: Optional[str]
    mode: str
    outcome: str  # scoped | low_risk | would_block | unguarded_mode_off


class EffectBoundary:
    def __init__(self, reassessment: Optional[ReassessmentEngine], *, mode_provider: Callable[[], str] = lambda: "observe",
                 eea: Iterable[str] = (), conditions: Any = None):
        self._reassessment = reassessment
        self._conditions = conditions  # ConditionManager: RUNTIME conditions are re-verified on every governed effect
        self._mode = mode_provider
        self._eea = frozenset(eea)
        self._lock = RLock()
        self._cov: Dict[str, Dict[str, int]] = {}
        self._targets: Dict[str, set] = {}

    # ------------------------------------------------------------------ guard
    def guard(self, kind: str, target: str, *, destination_country: Optional[str] = None, recipients: Sequence[str] = (),
              personal: Optional[bool] = None, scope: Optional[GovernedScope] = None) -> EffectPermit:
        if kind not in EFFECT_KINDS:
            raise ValueError(f"unknown effect kind {kind!r}; known: {sorted(EFFECT_KINDS)}")
        governed = EFFECT_KINDS[kind][0]
        scope = scope or _scope.get()
        mode = self._mode()
        self._count(kind, target, "observed")
        if scope is None:
            if not governed:
                self._count(kind, target, "low_risk")
                return EffectPermit(kind, target, None, mode, "low_risk")
            if mode == "enforce":
                self._count(kind, target, "denied")
                raise GovernanceDenied(self._deny(kind, target, "GOV-FX-01", "effect_outside_any_governed_scope"))
            self._count(kind, target, "unscoped_would_block")
            return EffectPermit(kind, target, None, mode, "would_block" if mode == "observe" else "unguarded_mode_off")
        # Scoped: inherit the authorisation, but prove it still holds *now* and that this effect stays inside it.
        try:
            if self._reassessment is not None and scope.auth_id:
                self._reassessment.check(scope.auth_id)
            elif governed and mode == "enforce" and scope.auth_id is None:
                raise AuthorizationRevoked("-", "missing", "scope carries no tracked authorisation")
        except AuthorizationRevoked:
            self._count(kind, target, "denied_lease")
            raise
        if governed and self._conditions is not None and scope.auth_id and mode != "off":
            try:
                self._conditions.runtime_gate(scope.auth_id, kind, target)
            except GovernanceDenied:
                self._count(kind, target, "denied_condition")
                if mode == "enforce":
                    raise
        if governed and scope.ctx is not None:
            reason = self._outside_authorised_scope(scope.ctx, destination_country, recipients, personal)
            if reason:
                self._count(kind, target, "denied_scope")
                raise GovernanceDenied(self._deny(kind, target, "GOV-FX-02", reason, scope))
        self._count(kind, target, "scoped")
        return EffectPermit(kind, target, scope.auth_id, mode, "scoped")

    def _outside_authorised_scope(self, ctx: ProcessingContext, destination: Optional[str], recipients: Sequence[str], personal: Optional[bool]) -> Optional[str]:
        from .processing_context import PERSONAL_DATA_CLASSES

        is_personal = personal if personal is not None else ctx.effective_data_class in PERSONAL_DATA_CLASSES
        dest = (destination or "").strip().upper()
        if dest and is_personal:
            authorised = (ctx.destination_country or "").strip().upper()
            if authorised and dest != authorised:
                return "effect_destination_differs_from_authorised_destination"
            if not authorised and dest not in self._eea:
                return "effect_destination_outside_eea_not_authorised"
        if recipients and is_personal and not set(map(str, recipients)) <= set(ctx.recipients):
            return "effect_recipient_not_authorised"
        return None

    def _deny(self, kind: str, target: str, rule: str, reason: str, scope: Optional[GovernedScope] = None) -> ProcessingDecision:
        return ProcessingDecision(scope.decision.request_id if scope else new_id("gov_req"), Decision.BLOCK,
                                  (RuleResult(rule, Decision.BLOCK, f"{reason}:{kind}", ("GDPR Art. 25", "GDPR Art. 32")),),
                                  scope.decision.policy_version if scope else "", scope.decision.context_fingerprint if scope else "")

    def _count(self, kind: str, target: str, what: str) -> None:
        with self._lock:
            row = self._cov.setdefault(kind, {})
            row[what] = row.get(what, 0) + 1
            targets = self._targets.setdefault(kind, set())
            if len(targets) < 200:
                targets.add(target)

    def coverage(self) -> Dict[str, Any]:
        """RUNTIME evidence: what the boundary actually saw in this process (counters reset on restart)."""
        with self._lock:
            kinds = {k: {**v, "targets": sorted(self._targets.get(k, ()))} for k, v in sorted(self._cov.items())}
        return {"basis": "runtime observation in this process; says nothing about code paths that never ran", "mode": self._mode(), "kinds": kinds}


# ---------------------------------------------------------------------- composition with the existing CommandGateway
class GovernanceCommandAuthority:
    """``Authority`` for ``CommandGateway`` that *wraps* the deployment's own authority.

    The inner authority still resolves identity and permissions (deny by default). Governance then checks the lease of the
    ambient scope and the effect's regulatory fit. A denial from either side is a denial; governance can only restrict.
    """

    def __init__(self, inner: Any, boundary: EffectBoundary, *, kind_for: Callable[[Any], str] = lambda capability: "command"):
        if not callable(getattr(inner, "authorize", None)):
            raise TypeError("inner authority must expose authorize(request_context, command, capability)")
        self._inner, self._boundary, self._kind_for = inner, boundary, kind_for

    def authorize(self, request_context: Any, command: Any, capability: Any) -> Any:
        decision = self._inner.authorize(request_context, command, capability)
        if not getattr(decision, "allowed", False):
            return decision
        effect = getattr(capability, "effect", "external")
        kind = "read_internal" if effect == "read" else self._kind_for(capability)
        try:
            self._boundary.guard(kind, str(getattr(command, "operation", "command")))
        except (GovernanceDenied, AuthorizationRevoked) as exc:
            return type(decision)(False, getattr(decision, "principal_id", ""), f"governance:{type(exc).__name__}")
        return decision


# ---------------------------------------------------------------------- process-wide installation (like the agent gate)
_lock = RLock()
_boundary: Optional[EffectBoundary] = None


def install_effect_boundary(boundary: Optional[EffectBoundary]) -> None:
    global _boundary
    with _lock:
        _boundary = boundary


def installed_effect_boundary() -> Optional[EffectBoundary]:
    with _lock:
        return _boundary


def guard_effect(kind: str, target: str, **kwargs: Any) -> Optional[EffectPermit]:
    """Call at a real side-effect site, immediately before acting. A no-op until the Governance Agent installs the boundary."""
    boundary = installed_effect_boundary()
    return boundary.guard(kind, target, **kwargs) if boundary is not None else None


# ---------------------------------------------------------------------- STATIC evidence
_SITE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "subprocess": ("subprocess.run", "subprocess.Popen", "subprocess.check_output", "subprocess.check_call", "subprocess.call", "os.system"),
    "network_outbound": ("requests.get", "requests.post", "requests.put", "requests.delete", "requests.request", "urlopen",
                         "urllib.request.urlopen", "httpx.get", "httpx.post"),
}


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def scan_effect_sites(root: Path, *, skip: Sequence[str] = ("governance", "tests", "__pycache__", "venv")) -> Dict[str, Any]:
    """STATIC evidence: every subprocess / outbound-network call site under ``root`` and whether the enclosing function also
    calls ``guard_effect`` (or composes ``GovernanceCommandAuthority``). It proves a guard is *written*, not that it *ran*;
    pair it with ``EffectBoundary.coverage()`` for runtime evidence."""
    sites: List[Dict[str, Any]] = []
    root = Path(root)
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in skip for part in rel.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            guarded = any(_dotted(c.func).endswith("guard_effect") for c in calls)
            for call in calls:
                name = _dotted(call.func)
                for kind, patterns in _SITE_PATTERNS.items():
                    if any(name == p or name.endswith("." + p) for p in patterns):
                        sites.append({"file": rel.as_posix(), "function": fn.name, "line": call.lineno, "kind": kind, "call": name, "guarded": guarded})
    # a function nested in another function is walked twice; keep one record per call site
    unique = {(s["file"], s["line"], s["call"]): s for s in sites}
    ordered = sorted(unique.values(), key=lambda s: (s["file"], s["line"]))
    unguarded = [s for s in ordered if not s["guarded"]]
    return {
        "basis": "static analysis of call sites; evidence that a guard is written next to the effect, not proof it executes",
        "sites": ordered,
        "summary": {"sites": len(ordered), "guarded": len(ordered) - len(unguarded), "unguarded": len(unguarded)},
        "unguarded": unguarded,
    }
