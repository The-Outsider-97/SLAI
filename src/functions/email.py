"""Email sending abstraction with SMTP support and async sending."""

from __future__ import annotations

import queue
import random
import smtplib
import socket
import threading
import time

from pathlib import Path
from email import encoders
from email.mime.base import MIMEBase
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from string import Template
from typing import Any, Callable, Dict, List, Optional, Union, Iterable
from enum import Enum

from .utils.config_loader import get_config_section
from .utils.functions_error import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Email Service")
printer = PrettyPrinter()


@dataclass(frozen=True)
class EmailRetryPolicy:
    """Retry settings for transient SMTP/network failures."""

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise EmailConfigurationError("retry_max must be >= 0")
        if self.base_delay < 0 or self.max_delay < 0:
            raise EmailConfigurationError("retry delays must be >= 0")
        if self.max_delay < self.base_delay:
            raise EmailConfigurationError("retry_max_delay must be >= retry_base_delay")
        if self.backoff_factor < 1:
            raise EmailConfigurationError("retry_backoff_factor must be >= 1")

    def sleep_duration(self, attempt: int) -> float:
        cap = min(self.max_delay, self.base_delay * (self.backoff_factor ** attempt))
        return random.uniform(0.0, cap) if self.jitter else cap


class EmailPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3

@dataclass
class EmailMessage:
    """Rich email message structure."""
    to: Union[str, List[str]]
    subject: str
    body_html: str
    body_text: Optional[str] = None
    from_addr: Optional[str] = None
    cc: Optional[Union[str, List[str]]] = None
    bcc: Optional[Union[str, List[str]]] = None
    reply_to: Optional[str] = None
    attachments: Optional[List[Attachment]] = None
    headers: Optional[Dict[str, str]] = None
    priority: EmailPriority = EmailPriority.NORMAL

@dataclass
class Attachment:
    filename: str
    content: Union[bytes, str]
    mime_type: Optional[str] = None  # e.g. "image/png", auto-detected if None

class EmailBackend(ABC):
    """Production-ready email backend abstraction."""

    @abstractmethod
    def send(self, message: EmailMessage) -> None:
        """Send a single email. Raise EmailError on failure."""
        pass

    def send_many(self, messages: Iterable[EmailMessage]) -> List[Exception]:
        """
        Send multiple emails. Returns list of exceptions for failed sends.
        Default implementation calls send() sequentially.
        """
        errors = []
        for msg in messages:
            try:
                self.send(msg)
            except Exception as e:
                errors.append(e)
        return errors

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify backend connectivity and credentials. Raise EmailError if not usable."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any persistent resources (e.g. connection pool)."""
        pass


class ConsoleBackend(EmailBackend):
    def send(self, message: EmailMessage) -> None:
        print(f"[CONSOLE EMAIL] To: {message.to}")
        print(f"Subject: {message.subject}")
        print(f"HTML: {message.body_html[:200]}...")
    def test_connection(self) -> bool: return True
    def close(self) -> None: pass


class SMTPBackend(EmailBackend):
    def __init__(
        self,
        host: str,
        port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        use_ssl: bool = False,          # For SMTPS (port 465)
        default_from: Optional[str] = None,
        timeout: float = 30.0,
        proxy: Optional[Dict[str, str]] = None,  # e.g. {"type": "socks5", "host": "...", "port": 1080}
        pool_size: int = 1,             # Simple per-thread pool
    ):
        if not isinstance(host, str) or not host.strip():
            raise EmailConfigurationError("SMTP host must be a non-empty string")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise EmailConfigurationError("SMTP port must be an integer within [1, 65535]")
        if timeout <= 0:
            raise EmailConfigurationError("SMTP timeout must be > 0")
        if pool_size != 1:
            raise EmailConfigurationError("SMTPBackend currently supports pool_size=1 only")
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.use_ssl = use_ssl
        self.default_from = default_from
        self.timeout = timeout
        self.proxy = proxy
        self._connection = None
        self._lock = threading.Lock()

    def _create_connection(self):
        """Create a new SMTP connection without process-global socket mutation."""
        if self.proxy:
            raise EmailConfigurationError(
                "SMTPBackend does not mutate process-global sockets; "
                "provide a dedicated proxy-aware EmailBackend instead"
            )
            socket.socket = socks.socksocket

        if self.use_ssl:
            server = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout)
        else:
            server = smtplib.SMTP(self.host, self.port, timeout=self.timeout)

        if self.use_tls and not self.use_ssl:
            server.starttls()

        if self.username and self.password:
            server.login(self.username, self.password)

        return server

    def _get_connection(self):
        with self._lock:
            if self._connection is None:
                self._connection = self._create_connection()
        return self._connection

    def _build_message(self, message: EmailMessage) -> MIMEMultipart:
        # Determine sender
        from_addr = message.from_addr or self.default_from
        if not from_addr:
            raise EmailConfigurationError("No sender address specified (missing from_addr and default_from)")

        recipients = self._normalize_recipients(message.to, "to")
        cc = self._normalize_recipients(message.cc, "cc") if message.cc else []
        self._validate_header("Subject", message.subject)
        self._validate_header("From", from_addr)

        msg = MIMEMultipart("mixed")
        msg["Subject"] = message.subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)

        if cc:
            msg["Cc"] = ", ".join(cc)
        if message.reply_to:
            self._validate_header("Reply-To", message.reply_to)
            msg["Reply-To"] = message.reply_to
        if message.headers:
            for k, v in message.headers.items():
                self._validate_header(str(k), str(v))
                msg[k] = v

        body = MIMEMultipart("alternative")
        if message.body_text:
            body.attach(MIMEText(message.body_text, "plain", "utf-8"))
        body.attach(MIMEText(message.body_html, "html", "utf-8"))
        msg.attach(body)

        # Attachments
        for att in (message.attachments or []):
            mime_type = att.mime_type or "application/octet-stream"
            main, sub = mime_type.split("/", 1) if "/" in mime_type else ("application", "octet-stream")
            part = MIMEBase(main, sub)

            # Normalize content to bytes (MIMEBase.set_payload accepts bytes or str)
            content = att.content
            if isinstance(content, memoryview):
                content = content.tobytes()
            if isinstance(content, str):
                content = content.encode("utf-8")
            # content is now bytes or bytearray
            part.set_payload(content)   # type: ignore[arg-type]  # safe because bytes
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{att.filename}"')
            msg.attach(part)

        return msg

    @staticmethod
    def _validate_header(name: str, value: str) -> None:
        if not isinstance(value, str) or "\r" in value or "\n" in value:
            raise EmailConfigurationError(f"Invalid {name} header value")

    @classmethod
    def _normalize_recipients(
        cls,
        value: Union[str, List[str]],
        field_name: str,
    ) -> List[str]:
        values = [value] if isinstance(value, str) else list(value)
        if not values:
            raise EmailConfigurationError(f"{field_name} must contain at least one address")
        for address in values:
            cls._validate_header(field_name, address)
            if not address.strip():
                raise EmailConfigurationError(f"{field_name} contains an empty address")
        return values

    def send(self, message: EmailMessage) -> None:
        recipients = []
        try:
            server = self._get_connection()
            msg = self._build_message(message)

            recipients.extend(self._normalize_recipients(message.to, "to"))
            if message.cc:
                recipients.extend(self._normalize_recipients(message.cc, "cc"))
            if message.bcc:
                recipients.extend(self._normalize_recipients(message.bcc, "bcc"))

            server.send_message(msg, to_addrs=recipients)
        except smtplib.SMTPAuthenticationError as e:
            raise EmailAuthError(f"SMTP authentication failed: {e}")
        except smtplib.SMTPRecipientsRefused as e:
            raise EmailSendError(recipient=", ".join(recipients), reason=str(e))
        except (smtplib.SMTPException, socket.error, OSError) as e:
            with self._lock:
                if self._connection:
                    try:
                        self._connection.quit()
                    except (smtplib.SMTPException, OSError):
                        pass
                    self._connection = None
            raise EmailSendError(recipient=", ".join(recipients), reason=str(e))

    def test_connection(self) -> bool:
        """Try to connect, authenticate, and quit."""
        try:
            conn = self._create_connection()
            conn.quit()
            return True
        except Exception:
            return False

    def close(self) -> None:
        with self._lock:
            if self._connection:
                try:
                    self._connection.quit()
                except Exception:
                    pass
                self._connection = None


class EmailService:
    """
    Email service with async queue and template rendering.

    Async queue backpressure
    ------------------------
    - queue_max_size bounds pending in-memory sends.
    - queue_put_timeout_seconds controls producer wait under pressure.
    - if queue_reject_on_full is True, sends fail fast on saturation.
    """

    def __init__(
        self,
        backend: EmailBackend,
        template_dir: Optional[str] = None,
        async_send: bool = True,
        queue_max_size: int = 1024,
        queue_put_timeout_seconds: float = 1.0,
        queue_reject_on_full: bool = True,
        retry_policy: Optional[EmailRetryPolicy] = None,
    ):
        if queue_max_size <= 0:
            raise ValueError("queue_max_size must be > 0")
        if queue_put_timeout_seconds < 0:
            raise ValueError("queue_put_timeout_seconds must be >= 0")

        self.backend = backend
        self.template_dir = template_dir
        self.async_send = async_send
        self.retry_policy = retry_policy or EmailRetryPolicy()
        self._queue = queue.Queue(maxsize=queue_max_size)
        self._queue_put_timeout_seconds = queue_put_timeout_seconds
        self._queue_reject_on_full = queue_reject_on_full
        self._worker: Optional[threading.Thread] = None

        if async_send:
            self._start_worker()

    def _execute_with_retry(self, operation: Callable[[], None]) -> None:
        last_exc: Exception | None = None
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                operation()
                return
            except EmailSendError as exc:
                last_exc = exc
                if attempt >= self.retry_policy.max_retries:
                    break
                delay = self.retry_policy.sleep_duration(attempt)
                logger.warning(
                    "Email send failed (attempt %s/%s); retrying in %.2fs: %s",
                    attempt + 1,
                    self.retry_policy.max_retries + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _start_worker(self) -> None:
        def worker() -> None:
            while True:
                try:
                    message = self._queue.get(timeout=1)
                except queue.Empty:
                    continue

                try:
                    if message is None:      # shutdown signal
                        return
                    self._execute_with_retry(lambda: self.backend.send(message)) # type: ignore
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                finally:
                    self._queue.task_done()

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a simple HTML template with string.Template."""
        if not self.template_dir:
            raise EmailConfigurationError("Template directory not configured")

        base_path = Path(self.template_dir).expanduser().resolve()
        path = (base_path / template_name).resolve()
        try:
            path.relative_to(base_path)
        except ValueError as exc:
            raise EmailTemplateError(template_name, "template path escapes template_dir") from exc

        try:
            with path.open("r", encoding="utf-8") as f:
                template = Template(f.read())
            return template.substitute(context)
        except (OSError, KeyError, ValueError) as exc:
            logger.error("Template rendering error for %s: %s", template_name, exc)
            raise EmailTemplateError(template_name, str(exc)) from exc

    def send(self,
             to: Union[str, List[str]],
             subject: str,
             body_html: str,
             body_text: Optional[str] = None,
             from_addr: Optional[str] = None,
             template: Optional[str] = None,
             context: Optional[Dict[str, Any]] = None,
             cc: Optional[Union[str, List[str]]] = None,
             bcc: Optional[Union[str, List[str]]] = None,
             reply_to: Optional[str] = None,
             attachments: Optional[List[Attachment]] = None,
             headers: Optional[Dict[str, str]] = None,
             priority: EmailPriority = EmailPriority.NORMAL,
             ) -> None:
        """
        Send an email. Legacy positional arguments supported.
        If `template` is given, render it with `context` to get body_html.
        """
        if template:
            if not context:
                context = {}
            body_html = self._render_template(template, context)

        # Build EmailMessage from parameters
        message = EmailMessage(
            to=to,
            subject=subject,
            body_html=body_html,
            body_text=body_text,
            from_addr=from_addr,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            attachments=attachments,
            headers=headers,
            priority=priority,
        )

        if self.async_send:
            # Put the message directly onto the queue
            try:
                self._queue.put(
                    message,
                    block=not self._queue_reject_on_full,
                    timeout=self._queue_put_timeout_seconds,
                )
            except queue.Full as exc:
                raise EmailError("Email queue is saturated; backpressure rejected send request") from exc
        else:
            self._execute_with_retry(lambda: self.backend.send(message))

    def shutdown(self) -> None:
        """Stop the worker thread gracefully."""
        if self._worker and self._worker.is_alive():
            self._queue.join()
            self._queue.put(None)
            self._worker.join(timeout=5)
        self.backend.close()

    @classmethod
    def from_config(cls, async_send: bool = True):
        config = get_config_section("email")
        backend_type = config.get("backend", "smtp")

        if backend_type == "smtp":
            # Ensure default_from is either a string or None
            default_from = config.get("default_from")
            if default_from is not None and not isinstance(default_from, str):
                default_from = str(default_from)   # defensive, though YAML should already be string

            backend = SMTPBackend(
                host=config["host"],
                port=config["port"],
                username=config.get("username", ""),
                password=config.get("password", ""),
                use_tls=config.get("use_tls", True),
                default_from=default_from,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")

        template_dir = config.get("template_dir")
        retry_policy = EmailRetryPolicy(
            max_retries=int(config.get("retry_max", 3)),
            base_delay=float(config.get("retry_base_delay", 0.5)),
            max_delay=float(config.get("retry_max_delay", 30.0)),
            backoff_factor=float(config.get("retry_backoff_factor", 2.0)),
            jitter=bool(config.get("retry_jitter", True)),
        )

        return cls(
            backend,
            template_dir,
            async_send,
            queue_max_size=int(config.get("queue_max_size", 1024)),
            queue_put_timeout_seconds=float(config.get("queue_put_timeout_seconds", 1.0)),
            queue_reject_on_full=bool(config.get("queue_reject_on_full", True)),
            retry_policy=retry_policy,
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "backend_healthy": self.backend.test_connection(),
            "queue_size": self._queue.qsize(),
            "retry_policy": {"max_retries": self.retry_policy.max_retries},
        }


__all__ = [
    "EmailRetryPolicy",
    "EmailPriority",
    "EmailMessage",
    "Attachment",
    "EmailBackend",
    "ConsoleBackend",
    "SMTPBackend",
    "EmailService",
]