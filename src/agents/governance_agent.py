"""Governance Agent: central regulatory authority for SLAI (GDPR/AVG, EU AI Act, UAVG).

This agent is the registered, lifecycle-managed face of ``governance.core.GovernanceCore``.
It owns regulatory *authority, applicability, orchestration and enforcement*; it does
not replace the specialist agents (Privacy, Safety, Alignment, Provenance, ...), which
keep ownership of their own detectors, state and evidence production.

Invariants
----------
* Enforcement is not optional. ``governance_agent.enabled: false`` is ignored (and
  logged at CRITICAL); there is no configuration that makes authorisation permissive.
* Missing dependencies degrade *readiness*, never the strictness of a decision:
  affected requests are QUARANTINED/BLOCKED (fail closed).
* The agent cannot certify itself: compliance statuses come from evidence produced
  by independent identities (Verification Agent / CI), see ``compliance_engine``.
* ``perform_task`` accepts only whitelisted operations; it is not a generic dispatcher.
"""

from __future__ import annotations

__version__ = "2.5.0"

import threading
import time

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .governance.compliance_engine import DeploymentProfile
from .governance.core import GovernanceCore
from .governance.processing_context import (
    DataClass,
    GovernanceDenied,
    Operation,
    ProcessingContext,
    ProcessingDecision,
)
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Governance Agent")
printer = PrettyPrinter()


class GovernanceAgent(BaseAgent):
    """Regulatory authority agent. See module docstring for invariants."""

    DEFAULT_STATE_DIR = "data/governance"

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.governance_config: Dict[str, Any] = dict(get_config_section("governance_agent", config=self.config) or {})
        if not self.governance_config:
            raise ValueError("Missing 'governance_agent' section in src/agents/base/configs/agents_config.yaml")

        if self.governance_config.get("enabled") is False:
            logger.critical(
                "governance_agent.enabled=false was configured and is IGNORED: regulatory enforcement is a mandatory "
                "architectural invariant and cannot be switched off by configuration."
            )
            self._mark_runtime_degraded("configuration", "governance.enabled_ignored", RuntimeError("attempted_disable"))

        state_dir = Path(self.governance_config.get("state_dir", self.DEFAULT_STATE_DIR))
        self.core = GovernanceCore.shared(state_dir, config=self.governance_config, **kwargs)
        self._register_agent_capabilities()
        self.profile_bootstrap = self.core.bootstrap_profiles(self.governance_config.get("ai_system_profiles", ()))
        if self.profile_bootstrap["errors"]:
            logger.error("Governance: %d configured AI-system profile(s) could not be registered", len(self.profile_bootstrap["errors"]))
        self._attach_privacy_classifier()
        self._install_delegation_authorizer()
        self.agent_gate = self._install_agent_gate()
        self._install_effect_boundary()
        self.runtime_registration = self._register_runtime()
        if self.governance_config.get("sanitize_logs", True):
            from .governance.utils.diagnostics import install_log_sanitizer

            install_log_sanitizer()
        health = self._report_core_health("initialize")
        logger.info("Governance Agent initialised | policy=%s | status=%s", health["policy_version"], health["status"])

    # ------------------------------------------------------------------ capability manifest
    def _register_agent_capabilities(self) -> None:
        """Register every factory agent's public actions as governed capabilities (static discovery)."""
        from .governance.agent_manifest import register_all

        counts = register_all(
            self.core.capabilities, Path(__file__).resolve().parent,
            personal_data_agents=self.governance_config.get("personal_data_agents", ()),
            human_approval_actions=self.governance_config.get("human_approval_actions", ()),
        )
        logger.info("Governance: %d agents / %d capabilities registered", len(counts), sum(counts.values()))

    def _attach_privacy_classifier(self) -> None:
        """Classification evidence comes from the Privacy Agent's ``DataID`` - the only sensitive-data detector in SLAI.

        The detector is constructed lazily on first use so that importing governance never drags in the Privacy stack; when it
        cannot be built, classification is *unavailable* and governance quarantines (it never guesses a class).
        """
        from .governance.classification_evidence import ClassifierUnavailable, PrivacyClassifierAdapter

        if not self.governance_config.get("classification", {}).get("use_privacy_agent", True):
            return
        holder: Dict[str, Any] = {}
        lock = threading.Lock()

        def classifier(payload: Any, *, purpose: str):
            adapter = holder.get("adapter")
            if adapter is None:
                with lock:
                    adapter = holder.get("adapter")
                    if adapter is None:
                        try:
                            from .privacy import DataID
                            from .privacy_agent import PrivacyAgent

                            adapter = PrivacyClassifierAdapter(DataID(), structural_check=PrivacyAgent._structurally_non_personal)
                        except Exception as exc:
                            raise ClassifierUnavailable(f"privacy classifier cannot be constructed: {type(exc).__name__}") from exc
                        holder["adapter"] = adapter
            return adapter(payload, purpose=purpose)

        self.core.classification.attach_classifier(classifier)

    def _install_effect_boundary(self) -> None:
        """Make the side-effect boundary process-wide, following the agent gate's mode (observe by default)."""
        from .governance.effect_boundary import install_effect_boundary

        install_effect_boundary(self.core.effects)

    def command_authority(self, inner: Any) -> Any:
        """Wrap the deployment's own ``CommandGateway`` authority: the inner decision is final for denials; governance can
        only restrict further (lease validity, destination, recipients)."""
        from .governance.effect_boundary import GovernanceCommandAuthority

        return GovernanceCommandAuthority(inner, self.core.effects)

    def integration_report(self) -> Dict[str, Any]:
        """Per-agent integration status. STATIC analysis (what is written) and RUNTIME observation (what ran) are reported
        separately and never merged: each carries its own ``basis``."""
        from .governance.agent_manifest import integration_report
        from .governance.effect_boundary import scan_effect_sites

        report = dict(integration_report(Path(__file__).resolve().parent, self.core.capabilities))
        report["effect_sites_static"] = scan_effect_sites(Path(__file__).resolve().parent)
        report["effect_boundary_runtime"] = self.core.effects.coverage()
        report["gate_runtime"] = self.agent_gate.status()
        report["coordination"] = dict(self.core.coordinator.stats)
        return report

    # ------------------------------------------------------------------ coordination
    def authorize_with_specialists(self, ctx, *, payload=None, requester=None):
        """Authorise ``ctx`` after consulting only the specialist agents whose findings could still change the outcome.

        Specialists are called through the governed delegation guard; findings can only restrict (see ``governance.coordination``).
        With no AgentFactory the required specialists cannot be reached and the request is QUARANTINED, never waved through.
        """
        from .governance.coordination import agent_invoker

        return self.core.authorize_coordinated(ctx, agent_invoker(self.agent_factory, self.shared_memory), payload=payload, requester=requester)

    # ------------------------------------------------------------------ delegation
    def _install_delegation_authorizer(self) -> None:
        """Wire the capability registry into the shared delegation guard (audit 4.6).

        ``enforce_capability_registry``:
          * ``true``   - every delegated ``agent.action`` must be a registered capability (denied otherwise);
          * ``"audit"``- (default) nothing is denied, but every delegation that WOULD be denied is counted and listed in
                         ``health()["delegation_audit"]`` - the evidence needed before switching to ``true``;
          * ``false``  - only the structural guard (private/dunder/lifecycle/injected actions are always refused).
        """
        from .governance.delegation_guard import DelegationDenied, install_authorizer

        setting = self.governance_config.get("enforce_capability_registry", "audit")
        self.delegation_audit = {"mode": "enforce" if setting is True else ("audit" if setting == "audit" else "off"), "checked": 0, "would_deny": []}
        if setting is not True and setting != "audit":
            install_authorizer(None)
            return
        registry = self.core.capabilities
        audit = self.delegation_audit

        def authorizer(caller: str, target: str, action: str) -> None:
            audit["checked"] += 1
            if registry.resolve(target, action) is None:
                if setting is True:
                    raise DelegationDenied(f"'{target}.{action}' is not a registered capability (GOV-CAP-13)")
                item = f"{target}.{action}"
                if item not in audit["would_deny"] and len(audit["would_deny"]) < 200:
                    audit["would_deny"].append(item)
                    logger.warning("Governance (audit): delegation %s -> %s would be denied (not a registered capability)", caller, item)

        install_authorizer(authorizer)

    # ------------------------------------------------------------------ central gate
    def _install_agent_gate(self):
        """Install the process-wide gate used by ``BaseAgent.execute`` (section 9). Default mode is observe, reported as such."""
        from .governance.agent_gate import AgentGate, install_gate
        from .governance.agent_manifest import factory_specs

        gc = dict(self.governance_config.get("agent_gate") or {})
        aliases = {spec.get("class_name"): key for key, spec in factory_specs(Path(__file__).resolve().parent).items() if spec.get("class_name")}
        gate = AgentGate(
            self.core, mode=str(gc.get("mode", "observe")), enforce_agents=gc.get("enforce_agents", ()),
            default_purpose=str(gc.get("default_purpose", "agent_execution")), default_purposes=gc.get("default_purposes") or {},
            default_tenant_id=gc.get("default_tenant_id"), learning_agents=gc.get("learning_agents", ("learning",)),
            learn_modes=gc.get("learn_modes", ("train", "auto", "fine_tune", "finetune")), marking_agents=gc.get("marking_agents") or {},
            aliases=aliases, # type: ignore
        )
        install_gate(gate)
        logger.info("Governance: agent gate installed | mode=%s", gate.mode)
        return gate

    def _register_runtime(self) -> Dict[str, Any]:
        """Register erasure adapters, access providers and the incident bridge for the stores reachable from here."""
        from .governance.registrations import register_runtime

        try:
            report = register_runtime(self.core, shared_memory=self.shared_memory, config=self.governance_config)
        except Exception as exc:  # never hide: degrade and report
            self._mark_runtime_degraded("registration", "register_runtime", exc)
            return {"error": type(exc).__name__, "not_covered": ["all stores: registration failed"]}
        return report

    # ------------------------------------------------------------------ readiness
    def _report_core_health(self, operation: str) -> Dict[str, Any]:
        health = self.core.health()
        if health["ready"]:
            self._mark_runtime_recovered("governance_core", operation)
        else:
            self._mark_runtime_degraded("governance_core", operation, RuntimeError(",".join(health["problems"])))
        return health

    def health(self) -> Dict[str, Any]:
        """Truthful health: includes runtime-contract status and the governance component checklist."""
        health = self._report_core_health("health")
        return {"agent": self.runtime_status(), "governance": health, "gate": self.agent_gate.status(),
                "delegation_audit": self.delegation_audit, "registration": self.runtime_registration}

    # ------------------------------------------------------------------ authority API
    def authorize(self, context: ProcessingContext) -> ProcessingDecision:
        return self.core.authorize(context)

    def enforce(self, context: ProcessingContext, *, accept_modifications: bool = False) -> ProcessingDecision:
        return self.core.enforce(context, accept_modifications=accept_modifications)

    def authorize_delegation(self, *, caller_agent: str, target_agent: str, action: str, context: ProcessingContext) -> ProcessingDecision:
        """Used by the Orchestrator/BaseAgent before any delegated call. Private/dunder actions are always refused."""
        if action.startswith("_"):
            # Never delegable, whatever the registry says: internals are not capabilities.
            from .governance.processing_context import Decision, RuleResult

            blocked = ProcessingDecision(
                request_id=context.request_id,
                decision=Decision.BLOCK,
                results=(RuleResult("GOV-CAP-13", Decision.BLOCK, "private_or_dunder_action_not_delegable", ("GDPR Art. 25",)),),
                policy_version=self.core.policies.active_version(),
                context_fingerprint=context.fingerprint(),
            )
            self.core._record(blocked)
            return blocked
        return self.core.authorize(self._with(context, capability=f"{target_agent}.{action}", operation=Operation.EXECUTE, agent_id=caller_agent))

    @staticmethod
    def _with(ctx: ProcessingContext, **changes: Any) -> ProcessingContext:
        data = {f: getattr(ctx, f) for f in ctx.__dataclass_fields__}
        data.update(changes)
        return ProcessingContext(**data)

    # ------------------------------------------------------------------ reporting
    def compliance_report(self, profile: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        deployment = DeploymentProfile(**dict(profile or self.governance_config.get("deployment_profile", {})))
        assessment = self.core.compliance.assess(deployment)
        return self.core.reporter.report(assessment)

    def release_gate(self, profile: Optional[Mapping[str, Any]] = None, *, include_organisational: bool = False):
        deployment = DeploymentProfile(**dict(profile or self.governance_config.get("deployment_profile", {})))
        return self.core.compliance.release_gate(deployment, include_organisational=include_organisational)

    # ------------------------------------------------------------------ BaseAgent contract
    _OPERATIONS = ("health", "compliance_report", "release_gate", "policy_version", "decision_log", "integration_report")

    def perform_task(self, task_data: Any) -> Any:
        operation = task_data.get("operation") if isinstance(task_data, Mapping) else None
        if operation not in self._OPERATIONS:
            raise PermissionError(
                f"GovernanceAgent.perform_task only supports {self._OPERATIONS}; "
                "use authorize()/enforce() with a ProcessingContext for authorisation."
            )
        if operation == "policy_version":
            return self.core.policies.active_version()
        if operation == "decision_log":
            return self.core.decision_log(int(task_data.get("limit", 50)))
        return getattr(self, operation)(**dict(task_data.get("params", {})))

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Shutdown the agent and close the core's owned resources."""
        logger.info("Governance Agent shutting down...")
        del timeout  # GovernanceCore.close() has no thread/background-task timeout parameter.
        self.core.close()
        logger.info("Governance Agent shutdown complete.")

    def __repr__(self) -> str:
        return f"<GovernanceAgent | policy={self.core.policies.active_version()} | status={self.runtime_status()}>"

    def __str__(self) -> str:
        return f"GovernanceAgent | policy={self.core.policies.active_version()} | status={self.runtime_status()}"

    def __enter__(self) -> GovernanceAgent:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()
        return None

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception as exc:
            logger.warning("Governance Agent shutdown failed in __del__: %s", exc)

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle support: only the config is preserved, the core is re-initialised on unpickle."""
        state = self.__dict__.copy()
        state["core"] = None
        return state