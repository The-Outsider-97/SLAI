"""
SLAI deployment factory for the R3D BIM Audit Platform.

Location
--------
SLAI/deployment/bimap.py

Architectural role
------------------
This module is the SLAI-owned deployment composition boundary. It resolves
concrete infrastructure and policy and returns the real
``applications.bimap.bootstrap.Bootstrap`` consumed by ``SLAI/bimap.py``.

Default behavior
----------------
The default ``development`` mode is intentionally self-contained and bootable:

- process-local Repository;
- process-local Storage;
- process-local Queue;
- UTC SystemClock;
- fail-closed Payment adapter;
- development malware gate;
- canonical BIMAP products without invented prices/tiers;
- valid API route hooks for local integration;
- explicit empty deterministic rule registry until product rules are supplied;
- explicit development Combined Audit version.

Production mode fails closed rather than silently using development adapters.
"""

from __future__ import annotations

import os

from collections.abc import Mapping
from typing import Any

from fastapi import Request

from applications.bimap.api.app import APISettings # type: ignore
from applications.bimap.api.dependencies import APIRouteHooks # type: ignore
from applications.bimap.api.middleware.request_limits import RequestLimitPolicy # type: ignore
from applications.bimap.api.middleware.security import SecurityPolicy # type: ignore
from applications.bimap.api.routes.downloads import DownloadGrant # type: ignore
from applications.bimap.api.utils.api_errors import APIServiceUnavailableError # type: ignore
from applications.bimap.audit_engine.bim_qa.auditor import BIMQAAuditor # type: ignore
from applications.bimap.audit_engine.combined.auditor import CombinedAuditor # type: ignore
from applications.bimap.audit_engine.rfa.auditor import RFAAuditor # type: ignore
from applications.bimap.audit_engine.rules.executor import RulesExecutor # type: ignore
from applications.bimap.audit_engine.rules.registry import RulesRegistry # type: ignore
from applications.bimap.bootstrap import ( # type: ignore
    Bootstrap,
    BootstrapAuditComponents,
    BootstrapConfiguration,
    BootstrapInfrastructure,
)
from applications.bimap.domain.products.models import ( # type: ignore
    ProductCatalog,
    ProductCode,
    ProductDefinition,
)
from applications.bimap.infra.local import ( # type: ignore
    DevelopmentMalware,
    DisabledPayment,
    InMemoryRepository,
    InMemoryStorage,
    InProcessQueue,
    SystemClock,
)

from logs.logger import PrettyPrinter, get_logger  # type: ignore
from src.agents.collaborative.shared_memory import SharedMemory  # type: ignore


logger = get_logger("BIMAP Deployment")
printer = PrettyPrinter()


_MODE_ENV = "BIMAP_ENV"
_COMBINED_VERSION_ENV = "BIMAP_COMBINED_AUDIT_VERSION"
_TRUST_UPLOADS_ENV = "BIMAP_DEV_TRUST_UPLOADS"
_ALLOWED_HOSTS_ENV = "BIMAP_ALLOWED_HOSTS"

_LOCAL_MODES = frozenset({"development", "dev", "local"})
_PRODUCTION_MODES = frozenset({"production", "prod"})


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _environment_mode() -> str:
    value = os.getenv(_MODE_ENV, "development").strip().casefold()

    if value in _LOCAL_MODES:
        return "development"

    if value in _PRODUCTION_MODES:
        return "production"

    raise RuntimeError(
        f"Unsupported {_MODE_ENV}={value!r}. "
        "Expected development/local or production."
    )


def _environment_bool(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    normalized = raw.strip().casefold()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(
        f"{name} must be a boolean environment value; received {raw!r}."
    )


def _allowed_hosts() -> tuple[str, ...]:
    raw = os.getenv(_ALLOWED_HOSTS_ENV)

    if raw is None:
        return ("127.0.0.1", "localhost")

    hosts: list[str] = []
    seen: set[str] = set()

    for item in raw.split(","):
        host = item.strip()

        if not host or host in seen:
            continue

        seen.add(host)
        hosts.append(host)

    if not hosts:
        raise RuntimeError(
            f"{_ALLOWED_HOSTS_ENV} must contain at least one hostname when set."
        )

    return tuple(hosts)


# ---------------------------------------------------------------------------
# Local HTTP/deployment hooks
# ---------------------------------------------------------------------------


async def _local_authorizer(
    request: Request,
    operation: str,
    resource_id: str | None,
) -> str:
    """Development-only route admission identity."""
    del request, operation, resource_id
    return "local-development"


async def _local_upload_manifest_validator(
    request: Request,
    order_id: str,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """
    Accept already-normalized manifest metadata in explicit local development.

    This hook does not claim to perform malware scanning or remote-object
    verification; those concerns remain separate infrastructure/application
    boundaries.
    """
    del request, order_id

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")

    return dict(manifest)


async def _local_report_id_resolver(
    request: Request,
    order_id: str,
) -> tuple[str, ...]:
    del request, order_id
    return ()


async def _local_download_url_issuer(
    request: Request,
    order_id: str,
    manifest: Any,
    artifact: Any,
) -> DownloadGrant:
    del request, order_id, manifest, artifact

    raise APIServiceUnavailableError(
        "Signed report downloads require a production object-storage adapter.",
        component="deployment_bimap",
        operation="issue_download_url",
    )


async def _local_deletion_admission_gate(
    request: Request,
    order_id: str,
    actor: str | None,
) -> None:
    del request, order_id, actor
    return None


async def _local_deletion_object_resolver(
    request: Request,
    order_id: str,
    actor: str | None,
) -> tuple[str, ...]:
    del request, order_id, actor
    return ()


def _build_route_hooks() -> APIRouteHooks:
    printer.status("BIMAP", "Building local API route hooks", "info")

    return APIRouteHooks(
        authorizer=_local_authorizer,
        upload_manifest_validator=_local_upload_manifest_validator,
        report_id_resolver=_local_report_id_resolver,
        download_url_issuer=_local_download_url_issuer,
        deletion_admission_gate=_local_deletion_admission_gate,
        deletion_object_resolver=_local_deletion_object_resolver,
        payment_signature_header="x-bimap-payment-signature",
    )


# ---------------------------------------------------------------------------
# Product configuration
# ---------------------------------------------------------------------------


def _build_catalog() -> ProductCatalog:
    """
    Build only canonical product identities already defined by BIMAP.

    No prices, currencies, commercial tiers, or limits are invented here.
    """
    printer.status("BIMAP", "Building canonical product catalog", "info")

    return ProductCatalog(
        products=tuple(
            ProductDefinition.canonical(product_code)
            for product_code in ProductCode
        ),
        tiers=(),
    )


# ---------------------------------------------------------------------------
# Deterministic Audit Engine composition
# ---------------------------------------------------------------------------


def _build_audit_components() -> BootstrapAuditComponents:
    """
    Compose the existing deterministic audit coordinators.

    The current repository contains the rule framework but no authoritative
    concrete RFA/BIM-QA rule implementations. An empty frozen registry is
    therefore preferable to fabricating audit policy. The service can boot and
    expose the complete architecture while returning no deterministic rule
    findings until real product rules are registered.
    """
    printer.status("BIMAP", "Building deterministic audit components", "info")

    registry = RulesRegistry()
    executor = RulesExecutor(registry)

    combined_version = os.getenv(
        _COMBINED_VERSION_ENV,
        "0.0.0",
    ).strip()

    if not combined_version:
        raise RuntimeError(
            f"{_COMBINED_VERSION_ENV} cannot be empty."
        )

    logger.warning(
        {
            "event": "bimap_development_audit_policy",
            "rule_count": len(registry),
            "combined_audit_version": combined_version,
            "message": (
                "No concrete deterministic product rules are currently "
                "registered by the repository."
            ),
        }
    )

    return BootstrapAuditComponents(
        rfa=RFAAuditor(executor),
        bim_qa=BIMQAAuditor(executor),
        combined=CombinedAuditor(combined_version),
    )


# ---------------------------------------------------------------------------
# API policy
# ---------------------------------------------------------------------------


def _build_api_settings() -> APISettings:
    printer.status("BIMAP", "Building local API settings", "info")

    return APISettings(
        request_limits=RequestLimitPolicy(),
        security=SecurityPolicy(
            allowed_hosts=_allowed_hosts(),
            require_https=False,
            add_nosniff=True,
            referrer_policy="no-referrer",
            frame_options="DENY",
            content_security_policy=(
                "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
            ),
            permissions_policy="camera=(), microphone=(), geolocation=()",
        ),
        api_prefix="/api/v1",
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )


# ---------------------------------------------------------------------------
# Deployment modes
# ---------------------------------------------------------------------------


def _create_local_bootstrap() -> Bootstrap:
    printer.status("BIMAP", "Constructing local BIMAP deployment", "info")

    trust_uploads = _environment_bool(
        _TRUST_UPLOADS_ENV,
        default=False,
    )

    if trust_uploads:
        logger.warning(
            {
                "event": "bimap_development_upload_trust_enabled",
                "environment": _TRUST_UPLOADS_ENV,
            }
        )

    infrastructure = BootstrapInfrastructure(
        repository=InMemoryRepository(),
        payment=DisabledPayment(),
        clock=SystemClock(),
        malware=DevelopmentMalware(
            trust_uploads=trust_uploads,
        ),
        storage=InMemoryStorage(),
        queue=InProcessQueue(),
        shared_memory=SharedMemory(),
        route_hooks=_build_route_hooks(),
        close_shared_memory_on_shutdown=True,
    )

    configuration = BootstrapConfiguration(
        catalog=_build_catalog(),
        api_settings=_build_api_settings(),
        product_limits=(),
        slai_profile=None,
        slai_required_agents=None,
        allow_degraded_slai_readiness=False,
        retain_slai_shared_memory=False,
        expose_health_details=False,
    )

    bootstrap = Bootstrap(
        infrastructure=infrastructure,
        configuration=configuration,
        audit_components=_build_audit_components(),
    )

    logger.info(
        {
            "event": "bimap_local_bootstrap_constructed",
            "deployment_mode": "development",
            "payment_enabled": False,
            "durable_persistence": False,
            "durable_queue": False,
            "trusted_upload_override": trust_uploads,
        }
    )

    return bootstrap


def _create_production_bootstrap() -> Bootstrap:
    """
    Refuse to misrepresent local adapters as production infrastructure.

    The production implementation must supply actual durable persistence,
    payment verification, malware scanning, object storage, queueing,
    authorization/download hooks, and authoritative deterministic audit policy.
    """
    raise RuntimeError(
        "BIMAP production mode is not configured with provider-backed adapters. "
        "Use BIMAP_ENV=development for the self-contained local deployment, "
        "or replace _create_production_bootstrap() with the deployment's real "
        "Repository, Payment, Malware, Storage, Queue, API hooks, and audit "
        "policy implementations."
    )


# ---------------------------------------------------------------------------
# Public deployment factory
# ---------------------------------------------------------------------------


def create_bootstrap() -> Bootstrap:
    """
    Construct one fully configured BIMAP Bootstrap instance.

    This is the zero-argument factory consumed by:

        SLAI/bimap.py

    No ``sys.path`` manipulation or hidden dependency discovery occurs here.
    """
    printer.status("BIMAP", "Constructing deployment bootstrap", "info")

    mode = _environment_mode()

    logger.info(
        {
            "event": "bimap_deployment_factory_start",
            "mode": mode,
        }
    )

    if mode == "production":
        return _create_production_bootstrap()

    return _create_local_bootstrap()


__all__ = [
    "create_bootstrap",
]
