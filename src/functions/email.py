"""Email sending abstraction with SMTP support and async sending."""

from __future__ import annotations

import threading
import queue
import smtplib

from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, Callable
from string import Template

from .utils.config_loader import get_config_section
from .utils.functions_error import EmailError
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Email Service")
printer = PrettyPrinter


class EmailBackend(ABC):
    """Abstract base class for email backends."""
    @abstractmethod
    def send(self, to: str, subject: str, body_html: str, body_text: Optional[str] = None) -> None:
        """Send an email. May raise EmailError."""
        pass


class SMTPBackend(EmailBackend):
    """SMTP email backend."""
    def __init__(self, host: str, port: int, username: str, password: str, use_tls: bool = True,
                 default_from: str = None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.default_from = default_from

    def send(self, to: str, subject: str, body_html: str, body_text: Optional[str] = None,
             from_addr: Optional[str] = None) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr or self.default_from
        msg["To"] = to

        if body_text:
            part_text = MIMEText(body_text, "plain")
            msg.attach(part_text)

        part_html = MIMEText(body_html, "html")
        msg.attach(part_html)

        try:
            with smtplib.SMTP(self.host, self.port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            logger.info(f"Email sent to {to}")
        except Exception as e:
            logger.error(f"Failed to send email to {to}: {e}")
            raise EmailError(f"SMTP send failed: {e}")


class EmailService:
    """
    Email service with async queue and template rendering.
    """
    def __init__(self, backend: EmailBackend, template_dir: Optional[str] = None,
                 async_send: bool = True):
        self.backend = backend
        self.template_dir = template_dir
        self.async_send = async_send
        self._queue = queue.Queue()
        self._worker = None
        if async_send:
            self._start_worker()

    def _start_worker(self) -> None:
        def worker():
            while True:
                try:
                    task = self._queue.get(timeout=1)
                    if task is None:   # poison pill
                        break
                    args, kwargs = task
                    self.backend.send(*args, **kwargs)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker error: {e}")

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a simple HTML template with string.Template."""
        if not self.template_dir:
            raise EmailError("Template directory not configured")
        import os
        path = os.path.join(self.template_dir, template_name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                template = Template(f.read())
            return template.safe_substitute(context)
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            raise EmailError(f"Template rendering failed: {e}")

    def send(self, to: str, subject: str, body_html: str, body_text: Optional[str] = None,
             from_addr: Optional[str] = None, template: Optional[str] = None,
             context: Optional[Dict[str, Any]] = None) -> None:
        """
        Send an email. If `template` is given, render it with `context` to get body_html.
        """
        if template:
            if not context:
                context = {}
            body_html = self._render_template(template, context)

        if self.async_send:
            self._queue.put(([to, subject, body_html, body_text, from_addr], {}))
        else:
            self.backend.send(to, subject, body_html, body_text, from_addr)

    def shutdown(self) -> None:
        """Stop the worker thread gracefully."""
        if self._worker and self._worker.is_alive():
            self._queue.put(None)   # poison pill
            self._worker.join(timeout=5)

    @classmethod
    def from_config(cls, async_send: bool = True):
        """Create EmailService from configuration section 'email'."""
        config = get_config_section('email')
        backend_type = config.get('backend', 'smtp')

        if backend_type == 'smtp':
            backend = SMTPBackend(
                host=config['host'],
                port=config['port'],
                username=config.get('username', ''),
                password=config.get('password', ''),
                use_tls=config.get('use_tls', True),
                default_from=config.get('default_from'),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")

        template_dir = config.get('template_dir')
        return cls(backend, template_dir, async_send)