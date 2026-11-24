"""Capability manifest and integration status for every factory agent (audit 4.6, section 9).

Derives each agent's delegable actions *statically* from its source (AST - no import, so no heavy runtime
needed and no side effects) and registers them as governed capabilities. Default posture is conservative:
every capability is limited to NON_PERSONAL / ANONYMISED data until an agent is explicitly listed in
``personal_data_agents`` (a business/legal decision, made in configuration - never inferred).

``integration_report`` states, per agent, what is actually true. The signals are static-analysis evidence of
integration, not proof of runtime behaviour, and are labelled as such.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .policy_registry import Capability, CapabilityRegistry
from .processing_context import DataClass

_PERSONAL_CLASSES = frozenset({DataClass.PERSONAL, DataClass.PSEUDONYMISED})


def _parse(path: Path) -> Optional[ast.Module]:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None


def factory_specs(agents_dir: Path) -> Dict[str, Dict[str, str]]:
    """Read ``DEFAULT_AGENT_SPECS`` from agent_factory.py without importing it."""
    tree = _parse(agents_dir / "agent_factory.py")
    if tree is None:
        return {}
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "DEFAULT_AGENT_SPECS":
            target = node.value
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_AGENT_SPECS" for t in node.targets):
            target = node.value
        if isinstance(target, ast.Dict):
            try:
                return {k: v for k, v in ast.literal_eval(target).items() if isinstance(v, dict)}
            except ValueError:
                return {}
    return {}


#: Public actions every agent inherits from ``BaseAgent`` (not visible in the subclass body, so static discovery would miss them).
INHERITED_ACTIONS = ("execute", "perform_task")


def public_actions(agents_dir: Path, spec: Mapping[str, str]) -> List[str]:
    module = spec.get("module_path", "").split(".")[-1]
    tree = _parse(agents_dir / f"{module}.py")
    if tree is None:
        return []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == spec.get("class_name"):
            names = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")}
            return sorted(names | set(INHERITED_ACTIONS))
    return []


def build_manifest(agents_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for agent, spec in sorted(factory_specs(agents_dir).items()):
        actions = public_actions(agents_dir, spec)
        out[agent] = {"class": spec.get("class_name"), "module": spec.get("module_path"), "actions": actions,
                      "resolved": bool(actions)}
    return out


def register_all(registry: CapabilityRegistry, agents_dir: Path, *, personal_data_agents: Iterable[str] = (),
                 human_approval_actions: Iterable[str] = ()) -> Dict[str, int]:
    """Register every discovered public action. Returns {agent: number registered}."""
    personal = set(personal_data_agents)
    approval = set(human_approval_actions)
    counts: Dict[str, int] = {}
    for agent, info in build_manifest(agents_dir).items():
        classes = frozenset({DataClass.NON_PERSONAL, DataClass.ANONYMISED}) | (_PERSONAL_CLASSES if agent in personal else frozenset())
        n = 0
        for action in info["actions"]:
            cid = f"{agent}.{action}"
            registry.register(Capability(capability_id=cid, owner_agent=agent, action=action, allowed_data_classes=classes,
                                         requires_human_approval=cid in approval, tenant_bound=agent in personal))
            n += 1
        counts[agent] = n
    return counts


def class_facts(agents_dir: Path, spec: Mapping[str, str]) -> Dict[str, bool]:
    """Static facts that decide whether ``BaseAgent.execute`` (and therefore the central gate) covers an agent."""
    module = spec.get("module_path", "").split(".")[-1]
    tree = _parse(agents_dir / f"{module}.py")
    facts = {"inherits_base_agent": False, "overrides_execute": False}
    if tree is None:
        return facts
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == spec.get("class_name"):
            bases = {getattr(x, "id", getattr(x, "attr", "")) for x in node.bases}
            facts["inherits_base_agent"] = "BaseAgent" in bases
            facts["overrides_execute"] = any(isinstance(n, ast.FunctionDef) and n.name == "execute" for n in node.body)
    return facts


_SIGNALS = {
    "privacy_gate": ("privacy_agent", "PrivacyAgent", "evaluate_privacy", "PrivacyDecision"),
    "governance_gate": ("governance_agent", "GovernanceAgent", ".authorize(", "enforce("),
    "runtime_status_contract": ("_mark_runtime_degraded", "runtime_status"),
}


def integration_report(agents_dir: Path, registry: Optional[CapabilityRegistry] = None) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    specs = factory_specs(agents_dir)
    for agent, info in build_manifest(agents_dir).items():
        facts = class_facts(agents_dir, specs.get(agent, {}))
        module = info["module"].split(".")[-1]
        try:
            text = (agents_dir / f"{module}.py").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        registered = sum(1 for a in info["actions"] if registry and registry.resolve(agent, a)) if registry else 0
        rows[agent] = {
            "source_resolved": info["resolved"],
            "capabilities_registered": registered,
            "delegation_guarded": True,  # BaseAgent.call_external and the orchestrator both use check_delegable
            "inherits_base_agent": facts["inherits_base_agent"],
            "overrides_execute": facts["overrides_execute"],
            # execute() is gated centrally (agent_gate); an agent is covered only if it inherits BaseAgent and does not replace execute()
            "covered_by_central_gate": facts["inherits_base_agent"] and not facts["overrides_execute"],
            "calls_privacy_gate": agent != "privacy" and any(s in text for s in _SIGNALS["privacy_gate"][2:]),
            "calls_governance_gate": agent != "governance" and any(s in text for s in _SIGNALS["governance_gate"][2:]),
            "own_gdpr_or_ai_act_engine": any(s in text for s in ("class GDPR", "class AIActChecker", "def check_gdpr")),
        }
    total = len(rows)
    return {
        "basis": "static analysis of source; evidence of integration, not proof of runtime behaviour",
        "agents": rows,
        "summary": {
            "agents": total,
            "capabilities_registered_for": sum(1 for r in rows.values() if r["capabilities_registered"]),
            "covered_by_central_gate": sum(1 for r in rows.values() if r["covered_by_central_gate"]),
            "not_covered_by_central_gate": sorted(a for a, r in rows.items() if not r["covered_by_central_gate"]),
            "calling_governance_gate": sum(1 for r in rows.values() if r["calls_governance_gate"]),
            "calling_privacy_gate": sum(1 for r in rows.values() if r["calls_privacy_gate"]),
            "with_own_compliance_engine": sorted(a for a, r in rows.items() if r["own_gdpr_or_ai_act_engine"]),
        },
    }
