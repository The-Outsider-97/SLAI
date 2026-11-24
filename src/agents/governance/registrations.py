"""Runtime registration of deletion adapters, rights providers and the incident bridge (audit 4.4, section 10).

``register_runtime`` is called by ``GovernanceAgent`` at start-up. It reports exactly what was registered and what
could not be, with the reason - a store that is not reachable is *reported as missing*, never silently skipped.
Idempotent: registrations are keyed by name.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from .deletion_adapters import (FileStoreDeletionAdapter, PrivacyMemoryDeletionAdapter, SecureMemoryDeletionAdapter,
                                SharedMemoryDeletionAdapter)
from .rights_providers import (FileStoreRightsProvider, PrivacyMemoryRightsProvider, SecureMemoryRightsProvider,
                               SharedMemoryRightsProvider)

logger = logging.getLogger(__name__)


def register_runtime(core: Any, *, shared_memory: Any = None, secure_memory: Any = None, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(config or {})
    registered: List[str] = []
    skipped: List[Dict[str, str]] = []
    portable = set(cfg.get("portable_stores", ()))  # Art. 20 eligibility is a deployment decision, never inferred

    def add(adapter: Any, provider: Any) -> None:
        core.ledger.register_adapter(adapter)
        core.rights.register_provider(provider)
        registered.append(adapter.name)

    def skip(name: str, reason: str) -> None:
        skipped.append({"store": name, "reason": reason})
        logger.warning("Governance: store '%s' is NOT covered by erasure/access (%s)", name, reason)

    # SharedMemory (the one agents already hold)
    if shared_memory is not None and callable(getattr(shared_memory, "get_by_tag", None)):
        add(SharedMemoryDeletionAdapter(shared_memory, persist_path=cfg.get("shared_memory_persist_path")),
            SharedMemoryRightsProvider(shared_memory, portable="shared_memory" in portable))
    else:
        skip("shared_memory", "no SharedMemory instance supplied")

    # SecureMemory (process-shared Safety evidence backend)
    sm = secure_memory
    if sm is None:
        try:
            from ..safety.secure_memory import SecureMemory

            sm = SecureMemory.shared()
        except Exception as exc:  # heavy/optional import: report, do not hide
            skip("secure_memory", f"unavailable: {type(exc).__name__}")
    if sm is not None:
        add(SecureMemoryDeletionAdapter(sm), SecureMemoryRightsProvider(sm, portable="secure_memory" in portable))

    # PrivacyMemory: one per privacy component; the adapter covers every live instance. PrivacyMemory is not tenant-aware,
    # so the deployment's single tenant must be named (never guessed).
    tenant = cfg.get("privacy_memory_tenant_id")
    if tenant:
        add(PrivacyMemoryDeletionAdapter(None, tenant_id=str(tenant)),
            PrivacyMemoryRightsProvider(tenant_id=str(tenant), portable="privacy_memory" in portable))
    else:
        skip("privacy_memory", "governance_agent.privacy_memory_tenant_id is not configured (PrivacyMemory is not tenant-aware)")

    for root in cfg.get("file_store_roots", ()) or ():
        name = "file_store:" + str(root)
        add(FileStoreDeletionAdapter(str(root), name=name), FileStoreRightsProvider(str(root), name=name, portable=name in portable))

    incident_attached = False
    if shared_memory is not None and callable(getattr(shared_memory, "subscribe", None)):
        try:
            core.incident_intake.attach(shared_memory, tuple(cfg.get("incident_channels", ("*:latest_incident",))))
            incident_attached = True
        except Exception as exc:
            skip("incident_bridge", f"attach failed: {type(exc).__name__}")
    else:
        skip("incident_bridge", "SharedMemory has no subscribe()")

    return {
        "deletion_adapters": core.ledger.adapter_names(),
        "rights_providers": core.rights.provider_names(),
        "registered_now": registered,
        "not_covered": skipped,
        "incident_bridge_attached": incident_attached,
        "known_uncovered_by_design": ["external recipients", "offline backups (tombstones only)", "model weights (unlearning assessment only)",
                                      "stores outside the registered set"],
    }
