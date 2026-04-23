"""
Human oversight orchestration for the alignment system.

Supports:
- Manual approval or rejection of model decisions.
- Injecting human preferences in ambiguous cases.
- Timeout and fallback handling for unattended requests.
- Structured intervention intake for high-risk alignment events.
- Multi-channel notification bookkeeping for dashboard/email/slack workflows.
- Persistent storage of pending requests (SQLite).
- Asynchronous non-blocking API.
- Actual channel integrations (Slack webhook, SMTP email).
- Authentication and role-based access control.

The module is designed to remain compatible with the broader alignment stack,
including the intervention flow used by AlignmentAgent._notify_humans().
"""

import asyncio
import hashlib
import json
import smtplib
import sqlite3
import sys
import threading
import uuid
import yaml # type: ignore

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from urllib.request import Request, urlopen
from urllib.error import URLError

from .config_loader import load_global_config, get_config_section
from .alignment_errors import HumanOversightValidationError, HumanOversightAuthError
from logs.logger import get_logger, PrettyPrinter # type: ignore

logger = get_logger("Human Oversight")
printer = PrettyPrinter

class HumanOversightTimeout(Exception):
    """Raised when a human oversight request times out."""
    def __init__(
        self,
        message: str = "Human oversight response timed out.",
        timeout_seconds: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        details = []
        if timeout_seconds is not None:
            details.append(f"timeout={timeout_seconds}s")
        if request_id:
            details.append(f"request_id={request_id}")
        if details:
            message = f"{message} ({', '.join(details)})"
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.request_id = request_id


@dataclass(frozen=True)
class OversightChannelResult:
    channel: str
    status: str
    delivered_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OversightRequest:
    request_id: str
    created_at: str
    urgency: str
    timeout_seconds: int
    channels: List[str]
    summary: Dict[str, Any]
    report: Dict[str, Any]
    status: str = "pending"  # pending, answered, expired, escalated
    escalation_level: int = 0


@dataclass(frozen=True)
class OversightResponse:
    request_id: str
    status: str
    decision: str
    feedback: Dict[str, Any]
    format: str
    responded_at: str
    reviewed_by: str
    channels: List[str]
    channel_delivery: List[Dict[str, Any]]


# ----------------------------------------------------------------------
# 1. Persistent Storage
# ----------------------------------------------------------------------
class OversightRequestStore:
    """SQLite-backed persistent store for oversight requests."""
    def __init__(self, db_path: str = "oversight_requests.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS oversight_requests (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    urgency TEXT,
                    timeout_seconds INTEGER,
                    channels TEXT,
                    summary TEXT,
                    report TEXT,
                    status TEXT,
                    escalation_level INTEGER,
                    last_notified_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS oversight_responses (
                    request_id TEXT PRIMARY KEY,
                    decision TEXT,
                    feedback TEXT,
                    format TEXT,
                    responded_at TEXT,
                    reviewed_by TEXT,
                    FOREIGN KEY(request_id) REFERENCES oversight_requests(request_id)
                )
            """)

    def save_request(self, req: OversightRequest):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO oversight_requests VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    req.request_id,
                    req.created_at,
                    req.urgency,
                    req.timeout_seconds,
                    json.dumps(req.channels),
                    json.dumps(req.summary),
                    json.dumps(req.report),
                    req.status,
                    req.escalation_level,
                    None,
                ),
            )

    def get_request(self, request_id: str) -> Optional[OversightRequest]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM oversight_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if not row:
                return None
            return OversightRequest(
                request_id=row[0],
                created_at=row[1],
                urgency=row[2],
                timeout_seconds=row[3],
                channels=json.loads(row[4]),
                summary=json.loads(row[5]),
                report=json.loads(row[6]),
                status=row[7],
                escalation_level=row[8],
            )

    def update_status(self, request_id: str, status: str, escalation_level: Optional[int] = None):
        with sqlite3.connect(self.db_path) as conn:
            if escalation_level is not None:
                conn.execute(
                    "UPDATE oversight_requests SET status = ?, escalation_level = ? WHERE request_id = ?",
                    (status, escalation_level, request_id),
                )
            else:
                conn.execute(
                    "UPDATE oversight_requests SET status = ? WHERE request_id = ?",
                    (status, request_id),
                )

    def save_response(self, resp: OversightResponse):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO oversight_responses VALUES (?,?,?,?,?,?)",
                (
                    resp.request_id,
                    resp.decision,
                    json.dumps(resp.feedback),
                    resp.format,
                    resp.responded_at,
                    resp.reviewed_by,
                ),
            )
            self.update_status(resp.request_id, "answered")

    def list_pending_requests(self) -> List[OversightRequest]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM oversight_requests WHERE status = 'pending' OR status = 'escalated'"
            ).fetchall()
            return [
                OversightRequest(
                    request_id=r[0],
                    created_at=r[1],
                    urgency=r[2],
                    timeout_seconds=r[3],
                    channels=json.loads(r[4]),
                    summary=json.loads(r[5]),
                    report=json.loads(r[6]),
                    status=r[7],
                    escalation_level=r[8],
                )
                for r in rows
            ]


# ----------------------------------------------------------------------
# 2. Channel Adapters (Actual Integrations)
# ----------------------------------------------------------------------
class ChannelAdapter:
    """Base class for notification channels."""
    def send(self, request: OversightRequest) -> OversightChannelResult:
        raise NotImplementedError

    @staticmethod
    def _payload_from_request(request: OversightRequest) -> Dict:
        return {
            "request_id": request.request_id,
            "urgency": request.urgency,
            "summary": request.summary,
            "report_preview": {k: v for k, v in request.report.items() if k != "full_data"},
        }


class SlackWebhookAdapter(ChannelAdapter):
    def __init__(self, webhook_url: str, retry_count: int = 3):
        self.webhook_url = webhook_url
        self.retry_count = retry_count

    def send(self, request: OversightRequest) -> OversightChannelResult:
        payload = self._payload_from_request(request)
        color = {"low": "#36a64f", "medium": "#ffcc00", "high": "#ff9900", "critical": "#ff0000"}.get(
            request.urgency, "#36a64f"
        )
        slack_message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"Human Intervention Required (Urgency: {request.urgency})",
                    "fields": [
                        {"title": "Request ID", "value": request.request_id, "short": True},
                        {"title": "Created At", "value": request.created_at, "short": True},
                        {"title": "Summary", "value": json.dumps(request.summary, indent=2), "short": False},
                    ],
                    "actions": [
                        {
                            "type": "button",
                            "text": "Respond via Dashboard",
                            "url": f"https://your-dashboard.com/oversight/{request.request_id}",
                        }
                    ],
                }
            ]
        }
        for attempt in range(self.retry_count):
            try:
                req = Request(self.webhook_url, data=json.dumps(slack_message).encode("utf-8"))
                req.add_header("Content-Type", "application/json")
                urlopen(req, timeout=10)
                return OversightChannelResult(
                    channel="slack",
                    status="delivered",
                    delivered_at=datetime.now().isoformat(),
                    metadata={"attempt": attempt + 1},
                )
            except URLError as e:
                logger.warning(f"Slack delivery attempt {attempt+1} failed: {e}")
                if attempt == self.retry_count - 1:
                    return OversightChannelResult(
                        channel="slack",
                        status="failed",
                        delivered_at=datetime.now().isoformat(),
                        metadata={"error": str(e)},
                    )
        # unreachable
        return OversightChannelResult(channel="slack", status="failed", delivered_at=datetime.now().isoformat())


class EmailAdapter(ChannelAdapter):
    def __init__(self, smtp_server: str, smtp_port: int = 587, username: str = None, password: str = None, from_addr: str = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr or "oversight@alignment.ai"

    def send(self, request: OversightRequest) -> OversightChannelResult:
        # Use a default recipient; in production this should come from config
        to_addr = "human-reviewer@example.com"
        subject = f"[{request.urgency.upper()}] Human Intervention Required: {request.request_id}"
        body = f"""
Request ID: {request.request_id}
Urgency: {request.urgency}
Created At: {request.created_at}
Summary: {json.dumps(request.summary, indent=2)}

Full report attached in the dashboard.
Please respond via the dashboard or use the CLI with request ID {request.request_id}.
"""
        msg = MIMEMultipart()
        msg["From"] = self.from_addr
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.username and self.password:
                    server.starttls()
                    server.login(self.username, self.password)
                server.send_message(msg)
            return OversightChannelResult(
                channel="email",
                status="delivered",
                delivered_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"Email delivery failed: {e}")
            return OversightChannelResult(
                channel="email",
                status="failed",
                delivered_at=datetime.now().isoformat(),
                metadata={"error": str(e)},
            )


class ConsoleAdapter(ChannelAdapter):
    def send(self, request: OversightRequest) -> OversightChannelResult:
        # Interactive console is handled separately; for notifications we just log
        logger.critical("Console intervention alert active for request %s", request.request_id)
        return OversightChannelResult(
            channel="console",
            status="interactive",
            delivered_at=datetime.now().isoformat(),
            metadata={"requires_input": True},
        )


class DashboardAdapter(ChannelAdapter):
    def send(self, request: OversightRequest) -> OversightChannelResult:
        # In production, this would push to a message queue or WebSocket
        logger.critical("Dashboard intervention alert: %s", json.dumps(self._payload_from_request(request), default=str))
        return OversightChannelResult(
            channel="dashboard",
            status="queued",
            delivered_at=datetime.now().isoformat(),
        )


# ----------------------------------------------------------------------
# 3. Authentication & Authorisation
# ----------------------------------------------------------------------
class AuthProvider:
    """Verifies reviewer identity and permissions."""
    def __init__(self, config: Dict):
        self.config = config.get("auth", {})
        self.api_keys = self.config.get("api_keys", {})  # mapping from key -> role
        self.role_permissions = self.config.get("role_permissions", {
            "reviewer": ["approve", "reject", "defer"],
            "senior_reviewer": ["approve", "reject", "defer", "escalate"],
            "admin": ["approve", "reject", "defer", "escalate", "override_config"],
        })

    def verify(self, token: str, required_decision: str) -> str:
        """Returns reviewer_id if authorised, raises HumanOversightAuthError."""
        if not token:
            raise HumanOversightAuthError("Missing authentication token.")
        # Check against hardcoded keys (in prod, use JWT or OAuth)
        role = self.api_keys.get(token)
        if not role:
            # Fallback: if no API keys defined, accept any non-empty token as "reviewer"
            if self.api_keys:
                raise HumanOversightAuthError("Invalid API key.")
            role = "reviewer"
        allowed_decisions = self.role_permissions.get(role, [])
        if required_decision not in allowed_decisions:
            raise HumanOversightAuthError(
                f"Role '{role}' is not authorised to make decision '{required_decision}'. "
                f"Allowed: {allowed_decisions}"
            )
        # Return a stable reviewer ID (could be derived from token)
        reviewer_id = f"{role}_{hashlib.sha256(token.encode()).hexdigest()[:8]}"
        return reviewer_id


# ----------------------------------------------------------------------
# 4. Main HumanOversightInterface (Enhanced)
# ----------------------------------------------------------------------
class HumanOversightInterface:
    """
    Enhanced interface with persistent storage, async API, channel integrations,
    and authentication.
    """

    VALID_DECISIONS = {"approve", "reject", "escalate", "defer"}

    def __init__(self, timeout_seconds: Optional[int] = None, operator_id: Optional[str] = None):
        self.oversight_config = get_config_section("human_oversight")
        # Timeout from args or config
        config_timeout = self.oversight_config.get("timeout_seconds")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else config_timeout
        if self.timeout_seconds is None:
            self.timeout_seconds = 30

        config_operator = self.oversight_config.get("operator_id")
        self.operator_id = operator_id if operator_id is not None else config_operator
        if self.operator_id is None:
            self.operator_id = "human_reviewer"

        # Persistent storage
        self.store = OversightRequestStore()

        # Channel adapters
        self._init_channel_adapters()

        # Auth provider
        self.auth = AuthProvider(self.oversight_config)

        # Internal state for blocking API (backward compatibility)
        self.response: Optional[str] = None
        self._lock = threading.Lock()
        self._audit_log: List[Dict[str, Any]] = []

        # Background task for escalation monitoring (async)
        self._escalation_task = None

    def _init_channel_adapters(self):
        self.channel_adapters = {}
        channels_cfg = self.oversight_config.get("channels", {})
        # Slack
        if "slack" in channels_cfg:
            slack_cfg = channels_cfg["slack"]
            webhook_url = slack_cfg.get("webhook_url")
            if webhook_url and webhook_url != "...":
                self.channel_adapters["slack"] = SlackWebhookAdapter(
                    webhook_url, retry_count=slack_cfg.get("retry_count", 3)
                )
        # Email
        if "email" in channels_cfg:
            email_cfg = channels_cfg["email"]
            smtp_server = email_cfg.get("smtp_server")
            if smtp_server and smtp_server != "...":
                self.channel_adapters["email"] = EmailAdapter(
                    smtp_server=smtp_server,
                    smtp_port=email_cfg.get("smtp_port", 587),
                    username=email_cfg.get("username"),
                    password=email_cfg.get("password"),
                    from_addr=email_cfg.get("from_addr"),
                )
        # Console and dashboard always available
        self.channel_adapters["console"] = ConsoleAdapter()
        self.channel_adapters["dashboard"] = DashboardAdapter()

    # ------------------------------------------------------------------
    # Async API (non-blocking)
    # ------------------------------------------------------------------
    async def request_intervention_async(
        self,
        report: Dict[str, Any],
        channels: Sequence[str],
        urgency: str,
        response_timeout: int = 300,
        auth_token: Optional[str] = None,
    ) -> OversightResponse:
        """
        Asynchronous version of request_intervention. Does not block the event loop.
        """
        request = self._build_request(report=report, channels=list(channels), urgency=urgency)
        # Override timeout if specified
        object.__setattr__(request, "timeout_seconds", response_timeout)
        self.store.save_request(request)

        # Dispatch notifications (non-blocking)
        deliveries = await self._dispatch_request_async(request)

        # Instead of blocking, we'll poll the store for a response.
        # We'll also start an escalation monitor in the background.
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < request.timeout_seconds:
            # Check if response has been stored
            resp = self._get_stored_response(request.request_id)
            if resp:
                return resp
            await asyncio.sleep(1)

        # Timeout reached
        self.store.update_status(request.request_id, "expired")
        raise HumanOversightTimeout(
            timeout_seconds=request.timeout_seconds,
            request_id=request.request_id,
        )

    async def _dispatch_request_async(self, request: OversightRequest) -> List[Dict[str, Any]]:
        deliveries = []
        for channel in request.channels:
            adapter = self.channel_adapters.get(channel)
            if not adapter:
                logger.warning(f"No adapter for channel '{channel}', skipping.")
                continue
            # Run blocking send in thread pool
            result = await asyncio.to_thread(adapter.send, request)
            deliveries.append(asdict(result))
        return deliveries

    def _get_stored_response(self, request_id: str) -> Optional[OversightResponse]:
        with sqlite3.connect(self.store.db_path) as conn:
            row = conn.execute(
                "SELECT decision, feedback, format, responded_at, reviewed_by FROM oversight_responses WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if not row:
                return None
            return OversightResponse(
                request_id=request_id,
                status="received",
                decision=row[0],
                feedback=json.loads(row[1]),
                format=row[2],
                responded_at=row[3],
                reviewed_by=row[4],
                channels=[],  # not stored in responses table, but can be retrieved from request
                channel_delivery=[],
            )

    # ------------------------------------------------------------------
    # Synchronous API (backward compatible, but now uses async under the hood)
    # ------------------------------------------------------------------
    @classmethod
    def request_intervention(
        cls,
        report: Dict[str, Any],
        channels: Sequence[str],
        urgency: str,
        response_timeout: int = 300,
        operator_id: str = "human_reviewer",
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous version that runs the async method in a temporary event loop.
        """
        interface = cls(timeout_seconds=response_timeout, operator_id=operator_id)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        response = loop.run_until_complete(
            interface.request_intervention_async(report, channels, urgency, response_timeout, auth_token)
        )
        return asdict(response)

    def request_approval(self, context: Dict[str, Any], auth_token: Optional[str] = None) -> bool:
        """
        Request human approval for a model decision. Authenticates reviewer.
        """
        prompt_lines = [
            "\n[HUMAN OVERSIGHT REQUIRED]",
            f"Context: {json.dumps(context, indent=2, default=str)}",
            "Please type 'approve' or 'reject'.",
        ]
        decision = self._prompt_with_timeout(
            prompt_lines=prompt_lines,
            parser=self._parse_binary_approval,
            timeout_seconds=self.timeout_seconds,
        )
        # Verify the decision is allowed for this reviewer (if token provided)
        if auth_token:
            reviewer_id = self.auth.verify(auth_token, decision)
            self.operator_id = reviewer_id
        return decision == "approve"

    def inject_preference(self, options: List[str], auth_token: Optional[str] = None) -> str:
        if not options:
            raise HumanOversightValidationError("Preference injection requires at least one option.")
        option_lines = ["\n[HUMAN PREFERENCE INJECTION]", "Choose the most ethically appropriate response:"]
        option_lines.extend(f"{idx}: {option}" for idx, option in enumerate(options))
        option_lines.append("Enter the number of your choice:")
        selected_index = self._prompt_with_timeout(
            prompt_lines=option_lines,
            parser=lambda raw: self._parse_preference_index(raw, len(options)),
            timeout_seconds=self.timeout_seconds,
        )
        # No authentication required for preference injection (can be extended)
        return options[selected_index]

    # ------------------------------------------------------------------
    # Helper methods (unchanged from original, but updated to use store)
    # ------------------------------------------------------------------
    def _build_request(self, report: Dict[str, Any], channels: List[str], urgency: str) -> OversightRequest:
        normalized_channels = self._normalize_channels(channels)
        request_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        return OversightRequest(
            request_id=request_id,
            created_at=created_at,
            urgency=(urgency or "high").lower(),
            timeout_seconds=int(self.timeout_seconds),
            channels=normalized_channels,
            summary=self._summarize_report(report),
            report=report or {},
        )

    def _summarize_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        metadata = report.get("metadata", {}) if isinstance(report, dict) else {}
        risk_analysis = report.get("risk_analysis", {}) if isinstance(report, dict) else {}
        recommendations = report.get("recommended_actions", {}) if isinstance(report, dict) else {}
        timeline = report.get("violation_timeline", []) if isinstance(report, dict) else []
        return {
            "agent_id": metadata.get("agent_id", "unknown"),
            "intervention_level": metadata.get("intervention_level", "UNKNOWN"),
            "protocol_version": metadata.get("protocol_version", "unknown"),
            "report_timestamp": metadata.get("report_timestamp", datetime.now().isoformat()),
            "total_risk": risk_analysis.get("total_risk", risk_analysis if isinstance(risk_analysis, (int, float)) else None),
            "component_risks": risk_analysis.get("component_risks", {}),
            "ethical_violations": risk_analysis.get("ethical_violations_details", []),
            "recommended_action_keys": list(recommendations.keys()) if isinstance(recommendations, dict) else [],
            "recent_violation_count": len(timeline) if isinstance(timeline, list) else 0,
        }

    def _normalize_channels(self, channels: Sequence[str]) -> List[str]:
        normalized = []
        for channel in channels:
            name = str(channel).strip().lower()
            if not name:
                continue
            if name not in self.channel_adapters:
                logger.warning("Unknown oversight channel '%s'; falling back to recorded delivery.", name)
            if name not in normalized:
                normalized.append(name)
        if "console" not in normalized:
            normalized.append("console")
        return normalized

    # ------------------------------------------------------------------
    # Interactive input (unchanged)
    # ------------------------------------------------------------------
    def _prompt_with_timeout(
        self,
        prompt_lines: List[str],
        parser: Callable[[str], Any],
        timeout_seconds: int,
        request_id: Optional[str] = None,
    ) -> Any:
        if not sys.stdin or not sys.stdin.isatty():
            raise HumanOversightTimeout(
                message="No interactive oversight channel is available for operator input.",
                timeout_seconds=timeout_seconds,
                request_id=request_id,
            )
        with self._lock:
            self.response = None
        response_event = threading.Event()
        error_holder: Dict[str, Exception] = {}

        def wait_for_input():
            try:
                raw_value = input("\n".join(prompt_lines)).strip()
                with self._lock:
                    self.response = raw_value
            except Exception as exc:
                error_holder["error"] = exc
            finally:
                response_event.set()

        input_thread = threading.Thread(target=wait_for_input, daemon=True)
        input_thread.start()
        if not response_event.wait(timeout=timeout_seconds):
            raise HumanOversightTimeout(timeout_seconds=timeout_seconds, request_id=request_id)
        if "error" in error_holder:
            raise error_holder["error"]
        with self._lock:
            raw_response = self.response or ""
        return parser(raw_response)

    def _parse_binary_approval(self, raw_response: str) -> str:
        decision = raw_response.strip().lower()
        if decision not in {"approve", "reject"}:
            raise HumanOversightValidationError("Approval response must be either 'approve' or 'reject'.")
        return decision

    def _parse_preference_index(self, raw_response: str, option_count: int) -> int:
        value = raw_response.strip()
        if not value.isdigit():
            raise HumanOversightValidationError("Preference selection must be a numeric index.")
        index = int(value)
        if index < 0 or index >= option_count:
            raise HumanOversightValidationError(f"Preference selection must be between 0 and {option_count - 1}.")
        return index

    # ------------------------------------------------------------------
    # Audit log (unchanged)
    # ------------------------------------------------------------------
    def get_audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)