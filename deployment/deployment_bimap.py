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
- explicit deterministic RFA baseline rule registry;
- grounded RFA rule-to-finding mapping policy;
- deployment-owned native Revit extraction backend;
- explicit development Combined Audit version.

Production mode fails closed rather than silently using development adapters.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from pathlib import Path
from collections.abc import Mapping
from typing import Any
from uuid import uuid4
from fastapi import Request # type: ignore

from applications.bimap.domain.accounts.models import Account  # type: ignore
from applications.bimap.domain.accounts.plans import AccountPlanCatalog, QuotaMode, UsageKind  # type: ignore
from applications.bimap.domain.orders.states import EXCEPTION_STATES, OrderState  # type: ignore
from applications.bimap.notifications import ContactMessageData, EmailAddress, EmailBranding, EmailRenderer, EmailService # type: ignore
from applications.bimap.notifications.providers import SMTPProvider # type: ignore
from applications.bimap.notifications.utils.email_errors import EmailValidationError # type: ignore
from applications.bimap.api.app import APISettings # type: ignore
from applications.bimap.api.dependencies import APIRouteHooks # type: ignore
from applications.bimap.api.middleware.request_limits import RequestLimitPolicy # type: ignore
from applications.bimap.api.middleware.security import SecurityPolicy # type: ignore
from applications.bimap.api.routes.auth import SESSION_COOKIE_NAME  # type: ignore
from applications.bimap.api.routes.downloads import DownloadGrant # type: ignore
from applications.bimap.api.utils.api_errors import APIServiceUnavailableError, APIUnauthorizedError  # type: ignore
from applications.bimap.app.utils.app_errors import AppValidationError # type: ignore
from applications.bimap.audit_engine.bim_qa.auditor import BIMQAAuditor # type: ignore
from applications.bimap.audit_engine.combined.auditor import CombinedAuditor # type: ignore
from applications.bimap.audit_engine.rfa.auditor import RFAAuditor # type: ignore
from applications.bimap.audit_engine.rfa.finding_mapper import map_rfa_finding # type: ignore
from applications.bimap.audit_engine.rfa.rules import RFA_RULES # type: ignore
from applications.bimap.audit_engine.rules.executor import RulesExecutor # type: ignore
from applications.bimap.audit_engine.rules.registry import RulesRegistry # type: ignore
from applications.bimap.bootstrap import ( # type: ignore
    Bootstrap,
    BootstrapAuditComponents,
    BootstrapConfiguration,
    BootstrapInfrastructure,
)
from applications.bimap.domain.products.models import ProductCatalog, ProductCode, ProductDefinition # type: ignore
from applications.bimap.infra.local import ( # type: ignore
    CalendarUTCRenewalWindowResolver,
    DevelopmentMalware,
    DisabledPayment,
    InMemoryAccounts,
    InMemoryEntitlementStore,
    InMemoryRepository,
    InMemoryStorage,
    InProcessQueue,
    LocalSLAIAuthentication,
    SystemClock,
)
from applications.bimap.infra.conversion import ( # type: ignore
    BlenderFbxModelConverter,
    IfcOpenShellModelConverter,
    MultiFormatModelConverter,
    RevitModelConverter,
    TrimeshModelConverter,
)
from applications.bimap.infra.email_artifact_mailer import EmailArtifactMailer  # type: ignore
from applications.bimap.infra.extraction import ( # type: ignore
    BlenderFbxDataExtractor,
    DwgDxfDataExtractor,
    IfcOpenShellDataExtractor,
    MultiFormatDataExtractor,
    RevitDataExtractor,
    PartAtomRfaDataExtractor,
    TrimeshDataExtractor,
)
from applications.bimap.infra.reportlab_data_extraction_renderer import ReportLabDataExtractionPDFRenderer  # type: ignore
from applications.bimap.infra.reportlab_audit_report_renderer import ReportLabAuditReportRenderer  # type: ignore
from applications.bimap.infra.sqlite_audit_results import SQLiteAuditResultStore # type: ignore
from applications.bimap.slai.task_builder import BIMAPSLAITaskBuilder # type: ignore
from applications.bimap.utils.plan_loader import load_account_plan_catalog # type: ignore
from applications.bimap.utils.config_loader import load_bimap_config, load_slai_profile  # type: ignore
from .revit_backend import RevitBackend, REVIT_EXTRACTOR_EXECUTABLE_ENV, create_revit_backends_from_env
from src.functions.auth import AuthService as SLAIAuthService  # type: ignore
from src.functions.phone_verification import PhoneVerificationService, TwilioBackend  # type: ignore
from src.agents.collaborative.shared_memory import SharedMemory  # type: ignore
from logs.logger import PrettyPrinter, get_logger  # type: ignore


logger = get_logger("BIMAP Deployment")
printer = PrettyPrinter()


_MODE_ENV = "BIMAP_ENV"
_COMBINED_VERSION_ENV = "BIMAP_COMBINED_AUDIT_VERSION"
_TRUST_UPLOADS_ENV = "BIMAP_DEV_TRUST_UPLOADS"
_ALLOWED_HOSTS_ENV = "BIMAP_ALLOWED_HOSTS"
_REQUIRE_SMS_VERIFICATION_ENV = "BIMAP_REQUIRE_SMS_VERIFICATION"
_SESSION_TTL_MINUTES_ENV = "BIMAP_SESSION_TTL_MINUTES"
_AUDIT_RESULTS_DB_ENV = "BIMAP_AUDIT_RESULTS_DB"
_LOCAL_MODES = frozenset({"development", "dev", "local"})
_PRODUCTION_MODES = frozenset({"production", "prod"})

def _build_email_service() -> EmailService:
    printer.status("BIMAP", "Building transactional email service","info")

    provider = SMTPProvider.from_env()
    renderer = EmailRenderer(
        EmailBranding(
            product_name="BIMAP",
            team_name="The Remy3Design Team",
            support_email="info.remy3design@gmail.com",
        )
    )

    return EmailService(
        renderer,
        provider,
        reply_to=EmailAddress(
            "info.remy3design@gmail.com",
            "Remy3Design",
        ),
    )

def _build_contact_message_sender(email_service: EmailService):
    """
    Adapt the concrete BIMAP EmailService to the API contact-message hook.

    Email address/content validation remains owned by the canonical
    notification models.
    """

    if not isinstance(email_service, EmailService):
        raise TypeError(
            "email_service must be an EmailService."
        )

    def _send(
        sender_name: str,
        sender_email: str,
        subject_display: str,
        message: str,
        ticket_number: str,
    ) -> None:
        try:
            data = ContactMessageData(
                sender_name=sender_name,
                sender_email=sender_email,
                subject_display=subject_display,
                message=message,
                ticket_number=ticket_number,
            )

            email_service.send_contact_message_email(
                data,
            )

        except EmailValidationError as exc:
            # This is request-data validation, not an internal
            # server failure. Translate it to the existing
            # application validation vocabulary so API
            # ErrorMapping returns a client error.
            raise AppValidationError(
                "Contact message validation failed.",
                component="deployment_bimap",
                operation="send_contact_message",
                field=getattr(
                    exc,
                    "field",
                    None,
                ),
                cause=exc,
            ) from exc

    return _send

def _required_environment(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Required environment variable {name} is not configured.")

    return value.strip()


def _build_phone_verification_service() -> PhoneVerificationService:
    printer.status("BIMAP", "Building SMS verification service", "info")

    backend = TwilioBackend(
        account_sid=_required_environment("BIMAP_TWILIO_ACCOUNT_SID"),
        auth_token=_required_environment("BIMAP_TWILIO_AUTH_TOKEN"),
        from_number=_required_environment("BIMAP_TWILIO_FROM_NUMBER"),
    )

    if not backend.test_connection():
        raise RuntimeError("Twilio SMS authentication or connectivity test failed.")

    return PhoneVerificationService(
        backend,
        code_length=6,
        code_ttl_seconds=300.0,
        max_verify_attempts=5,
        resend_cooldown_seconds=60.0,
    )


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

    raise RuntimeError(f"{name} must be a boolean environment value; received {raw!r}.")


def _environment_positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)

    if raw is None:
        return default

    try:
        value = int(raw.strip())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")

    return value


def _allowed_hosts() -> tuple[str, ...]:
    raw = os.getenv(_ALLOWED_HOSTS_ENV)

    if raw is None:
        config = (load_bimap_config())

        return tuple(config["api"]["security"]["allowed_hosts"])

    hosts: list[str] = []
    seen: set[str] = set()

    for item in raw.split(","):
        host = item.strip()

        if (
            not host
            or host in seen
        ):
            continue

        seen.add(host)
        hosts.append(host)

    if not hosts:
        raise RuntimeError(
            f"{_ALLOWED_HOSTS_ENV} "
            "must contain at least one "
            "hostname when set."
        )

    return tuple(hosts)

def _audit_results_database_path() -> Path:
    """
    Resolve durable BIMAP audit-workspace storage.

    An explicit environment value always wins. For development, the default
    resolves beneath the SLAI working directory so local audit workspaces
    survive backend restarts.
    """

    configured = os.getenv(_AUDIT_RESULTS_DB_ENV)

    if (
        configured is not None
        and configured.strip()
    ):
        return (Path(configured.strip()).expanduser().resolve(strict=False))

    default_path = (
        Path.cwd()
        / "data"
        / "bimap"
        / "audit_results.sqlite3"
    ).resolve(strict=False)

    logger.info(
        {
            "event": "bimap_default_audit_result_database",
            "environment": _AUDIT_RESULTS_DB_ENV,
            "configured": False,
        }
    )

    return default_path


# ---------------------------------------------------------------------------
# Local HTTP/deployment hooks
# ---------------------------------------------------------------------------

def _local_authorizer_factory(authentication, accounts):
    async def _local_authorizer(request: Request, operation: str, resource_id: str | None) -> str:
        del operation, resource_id
        raw = request.cookies.get(SESSION_COOKIE_NAME)
        if not raw or not raw.strip():
            raise APIUnauthorizedError(
                "BIMAP authentication session is required.",
                component="deployment_bimap",
                operation="authorize_request",
            )

        principal = authentication.resolve_session(raw.strip())
        if principal is None:
            raise APIUnauthorizedError(
                "BIMAP authentication session is invalid or expired.",
                component="deployment_bimap",
                operation="authorize_request",
            )

        account = accounts.get_by_auth_user_id(principal.auth_user_id)
        if account is None or not account.can_authenticate:
            raise APIUnauthorizedError(
                "BIMAP authentication session has no active account.",
                component="deployment_bimap",
                operation="authorize_request",
            )

        return account.account_id

    return _local_authorizer


def _build_route_hooks(authentication, accounts, contact_message_sender) -> APIRouteHooks:
    printer.status("BIMAP", "Building local API route hooks", "info")
    return APIRouteHooks(
        authorizer=_local_authorizer_factory(authentication, accounts),
        upload_manifest_validator=_local_upload_manifest_validator,
        report_id_resolver=_local_report_id_resolver,
        download_url_issuer=_local_download_url_issuer,
        deletion_admission_gate=_local_deletion_admission_gate,
        deletion_object_resolver=_local_deletion_object_resolver,
        payment_signature_header="x-bimap-payment-signature",
        contact_message_sender=contact_message_sender,
    )


def _format_utc(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _build_local_account_summary_resolver(
    *,
    repository,
    entitlement_store,
    renewal_window_resolver,
    clock,
    plan_catalog: AccountPlanCatalog,
):
    async def _resolve(request: Request, account: Account):
        del request
        plan = plan_catalog.get(account.plan_code)
        now = clock.now()

        def usage(kind: UsageKind) -> dict[str, Any]:
            quota = plan.quota_for(kind)
            bonus = entitlement_store.bonus_balance(
                account_id=account.account_id,
                kind=kind,
            )

            if quota.mode is QuotaMode.UNLIMITED:
                return {
                    "used": None,
                    "limit": None,
                    "remaining": None,
                    "unlimited": True,
                    "renewal": quota.renewal.value,
                    "periodStart": None,
                    "periodEnd": None,
                    "bonusCredits": bonus,
                }

            window = renewal_window_resolver.resolve(
                account_id=account.account_id,
                plan=plan,
                quota=quota,
                at=now,
            )
            used = entitlement_store.recurring_usage_count(
                account_id=account.account_id,
                kind=kind,
                period_start=window.start,
                period_end=window.end,
            )
            assert quota.limit is not None
            return {
                "used": used,
                "limit": quota.limit,
                "remaining": max(quota.limit - used, 0),
                "unlimited": False,
                "renewal": quota.renewal.value,
                "periodStart": _format_utc(window.start),
                "periodEnd": _format_utc(window.end),
                "bonusCredits": bonus,
            }

        audits = {
            "inProgress": [],
            "done": [],
            "cancelled": [],
        }
        for order in repository.list_orders_for_account(account.account_id):
            if order.state is OrderState.DELIVERED:
                bucket = "done"
                public_status = "done"
            elif order.state in EXCEPTION_STATES:
                bucket = "cancelled"
                public_status = "cancelled"
            else:
                bucket = "inProgress"
                public_status = "in_progress"

            audits[bucket].append(
                {
                    "id": order.order_id,
                    "product": order.product_code,
                    "projectAlias": order.project_alias,
                    "status": public_status,
                    "updatedAt": _format_utc(order.updated_at),
                }
            )

        return {
            "audits": audits,
            "purchases": {"threeD": [], "twoD": []},
            "currentPlan": account.plan_code.value,
            "usage": {
                "audit": usage(UsageKind.AUDIT),
                "conversion": usage(UsageKind.CONVERSION),
                "dataExtraction": usage(UsageKind.DATA_EXTRACTION),
            },
            "rewards": {
                "points": 0,
                "basePurchaseDiscountPercent": float(plan.base_purchase_discount_percent),
                "maxEffectivePurchaseDiscountPercent": float(plan.max_effective_purchase_discount_percent),
                "maxRewardDiscountPercent": float(plan.max_reward_discount_percent),
            },
        }

    return _resolve


async def _local_upload_manifest_validator(request: Request, order_id: str, manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
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


async def _local_report_id_resolver(request: Request, order_id: str) -> tuple[str, ...]:
    del request, order_id
    return ()


async def _local_download_url_issuer(request: Request, order_id: str, manifest: Any, artifact: Any) -> DownloadGrant:
    del request, order_id, manifest, artifact

    raise APIServiceUnavailableError(
        "Signed report downloads require a production object-storage adapter.",
        component="deployment_bimap",
        operation="issue_download_url",
    )


async def _local_deletion_admission_gate(request: Request, order_id: str, actor: str | None) -> None:
    del request, order_id, actor
    return None


async def _local_deletion_object_resolver(request: Request, order_id: str, actor: str | None) -> tuple[str, ...]:
    del request, order_id, actor
    return ()


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
    Compose deterministic BIMAP audit coordinators with the approved
    repository-grounded RFA baseline rules and explicit finding policy.

    Product rules are registered before RulesExecutor construction because
    RulesExecutor freezes the registry for deterministic execution.
    """
    printer.status("BIMAP", "Building deterministic audit components", "info")

    registry = RulesRegistry(RFA_RULES)
    executor = RulesExecutor(registry)
    combined_version = os.getenv(_COMBINED_VERSION_ENV, "0.0.0").strip()

    if not combined_version:
        raise RuntimeError(f"{_COMBINED_VERSION_ENV} cannot be empty." )

    logger.info(
        {
            "event": "bimap_deterministic_audit_policy",
            "rule_count": len(registry),
            "rfa_rule_count": len(RFA_RULES),
            "combined_audit_version": combined_version,
            "rfa_finding_mapper": "configured",
        }
    )

    return BootstrapAuditComponents(
        rfa=RFAAuditor(executor, finding_mapper=map_rfa_finding),
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

    trust_uploads = _environment_bool(_TRUST_UPLOADS_ENV, default=True)

    if trust_uploads:
        logger.warning(
            {
                "event": "bimap_development_upload_trust_enabled",
                "environment": _TRUST_UPLOADS_ENV,
            }
        )

    plan_catalog = load_account_plan_catalog()
    clock = SystemClock()
    repository = InMemoryRepository()
    accounts = InMemoryAccounts()
    audit_results = SQLiteAuditResultStore(_audit_results_database_path())

    # ---------------------------------------------------------
    # Local account-entitlement infrastructure
    # ---------------------------------------------------------

    entitlement_store = InMemoryEntitlementStore()
    renewal_window_resolver = CalendarUTCRenewalWindowResolver()
    def _auth_memory_path() -> Path:
        configured = os.getenv("BIMAP_AUTH_MEMORY_PATH")

        if (
            configured is not None
            and configured.strip()
        ):
            path = (Path(configured.strip()).expanduser().resolve(strict=False))
        else:
            path = (
                Path.cwd()
                / "data"
                / "bimap"
                / "auth.json"
            ).resolve(strict=False)

        path.parent.mkdir(parents=True, exist_ok=True,)

        return path
    email_notifications = _build_email_service()
    artifact_mailer = EmailArtifactMailer(email_notifications)
    require_sms_verification = _environment_bool(_REQUIRE_SMS_VERIFICATION_ENV, default=False)

    phone_verification = (
        _build_phone_verification_service()
        if require_sms_verification
        else None
    )

    session_ttl_minutes = _environment_positive_int(_SESSION_TTL_MINUTES_ENV, default=480,)
    auth_memory_path = str(_auth_memory_path())
    authentication = LocalSLAIAuthentication(
        SLAIAuthService(memory_path=auth_memory_path, token_ttl_minutes=session_ttl_minutes),
        email_notifications,
        phone_verification,
        email_code_ttl_minutes=15,
        require_phone_verification=require_sms_verification,
    )

    account_summary_resolver = _build_local_account_summary_resolver(
        repository=repository,
        entitlement_store=entitlement_store,
        renewal_window_resolver=renewal_window_resolver,
        clock=clock,
        plan_catalog=plan_catalog,
    )

    mesh_extractor = TrimeshDataExtractor()
    conversion_adapters = [IfcOpenShellModelConverter(), TrimeshModelConverter()]
    extraction_adapters = [
        IfcOpenShellDataExtractor(),
        mesh_extractor,
        # With no DWG backend this adapter advertises DXF, not DWG.
        DwgDxfDataExtractor(),
    ]

    revit_executable = os.getenv(REVIT_EXTRACTOR_EXECUTABLE_ENV)

    if (
        revit_executable is not None
        and revit_executable.strip()
    ):
        configured_revit_worker = (Path(revit_executable.strip()).expanduser())

        if (
            configured_revit_worker
            .suffix
            .casefold()
            in {
                ".dll",
                ".addin",
            }
        ):
            raise RuntimeError(
                f"{REVIT_EXTRACTOR_EXECUTABLE_ENV} "
                "must point to a standalone native Revit worker executable "
                "implementing BIMAP's inspect/extract/convert/preview CLI contract. "
                "native/revit_local_exporter/Bimap.RevitLocalExporter.dll is a "
                "Revit IExternalCommand add-in and must not be registered as that "
                "process worker. Its generated GLB must instead enter BIMAP through "
                "/orders/{order_id}/uploads/viewer-model."
            )

        (
            revit_extraction_backend,
            revit_conversion_backend,
        ) = create_revit_backends_from_env()

        extraction_adapters.append(RevitDataExtractor(revit_extraction_backend))
        conversion_adapters.append(RevitModelConverter(revit_conversion_backend))

        logger.info(
            {
                "event":
                    "bimap_revit_processing_enabled",
                "formats": ("rfa", "rvt"),
                "conversion_targets": ("glb",),
            }
        )

    else:
        # THIS LINE IS CURRENTLY EFFECTIVELY MISSING
        # FROM THE DEPLOYMENT THAT PRODUCED YOUR LOG.
        extraction_adapters.append(
            PartAtomRfaDataExtractor()
        )

        logger.warning(
            {
                "event": "bimap_revit_partatom_fallback_enabled",
                "reason": "native_revit_worker_not_configured",
                "environment": REVIT_EXTRACTOR_EXECUTABLE_ENV,
                "limitations": (
                    "no_formulas",
                    "no_connectors",
                    "no_nested_topology",
                    "no_native_geometry",
                    "no_glb_conversion",
                ),
            }
        )


    blender = shutil.which("blender")

    if blender is not None:
        conversion_adapters.append(BlenderFbxModelConverter(blender_executable=blender))
        extraction_adapters.append(BlenderFbxDataExtractor(blender_executable=blender, mesh_extractor=mesh_extractor))

    model_converter = MultiFormatModelConverter(*conversion_adapters)
    data_extractor = MultiFormatDataExtractor(*extraction_adapters)
    supported_extraction_extensions = {
        extension.casefold()
        for capability
        in data_extractor.capabilities
        for extension
        in capability.extensions
    }
    if ".rfa" not in supported_extraction_extensions:
        raise RuntimeError(
            "Family Audit is configured but no "
            "RFA data extractor is registered."
        )
    data_extraction_pdf_renderer = ReportLabDataExtractionPDFRenderer()
    audit_report_renderer = ReportLabAuditReportRenderer()

    # ---------------------------------------------------------
    # Local infrastructure
    # ---------------------------------------------------------

    infrastructure = BootstrapInfrastructure(
        repository=repository,
        payment=DisabledPayment(),
        clock=clock,
        malware=DevelopmentMalware(trust_uploads=trust_uploads),
        storage=InMemoryStorage(),
        queue=InProcessQueue(),
        accounts=accounts,
        authentication=authentication,
        entitlement_store=entitlement_store,
        renewal_window_resolver=renewal_window_resolver,
        shared_memory=SharedMemory(),
        route_hooks=_build_route_hooks(authentication, accounts, _build_contact_message_sender(email_notifications)),
        notifications=email_notifications,
        artifact_mailer=artifact_mailer,
        account_summary_resolver=account_summary_resolver,
        account_avatar_uploader=None,
        close_shared_memory_on_shutdown=True,
        model_converter=model_converter,
        data_extractor=data_extractor,
        data_extraction_pdf_renderer=data_extraction_pdf_renderer,
        report_renderer=audit_report_renderer,
        audit_results=audit_results,
        slai_task_builder=BIMAPSLAITaskBuilder(),
    )

    # ---------------------------------------------------------
    # Validated BIMAP configuration
    # ---------------------------------------------------------

    configuration = BootstrapConfiguration(
        catalog=_build_catalog(),
        api_settings=_build_api_settings(),
        account_plans=plan_catalog,
        product_limits=(),
        slai_profile=load_slai_profile(),
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
            "authentication_enabled": True,
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
