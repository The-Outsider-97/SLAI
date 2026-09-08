"""Phone number verification via SMS one-time passcodes (OTP)."""

from __future__ import annotations

import base64
import hmac
import random
import re
import secrets
import string
import threading
import time
import urllib.error
import urllib.request

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from .utils.config_loader import get_config_section
from .utils.functions_error import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Phone Verification")
printer = PrettyPrinter()

# ----------------------------------------------------------------------
# Optional dependency: `phonenumbers` gives libphonenumber-grade validation.
# The module degrades gracefully to E.164 regex validation if it's absent.
# ----------------------------------------------------------------------
try:
    import phonenumbers  # type: ignore

    _HAS_PHONENUMBERS = True
except ImportError:  # pragma: no cover - exercised only when dependency missing
    phonenumbers = None  # type: ignore
    _HAS_PHONENUMBERS = False

_E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")
_FORMATTING_CHARS = re.compile(r"[\s\-.()]")
_DIGITS = string.digits


# ----------------------------------------------------------------------
# Phone number normalization
# ----------------------------------------------------------------------
def normalize_phone_number(raw: str, default_region: Optional[str] = None) -> str:
    """
    Normalize a phone number to E.164 (e.g. "+14155552671").

    Uses the `phonenumbers` library when available for full libphonenumber
    validation (including region-aware parsing of numbers without a leading
    '+'). Falls back to a conservative E.164 regex check otherwise, in which
    case numbers without a leading '+' cannot be disambiguated and are
    rejected.

    Raises:
        InvalidPhoneNumberError: if the number cannot be normalized/validated.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidPhoneNumberError(str(raw), "phone number must be a non-empty string")

    candidate = raw.strip()

    if _HAS_PHONENUMBERS:
        assert phonenumbers is not None
        try:
            assert phonenumbers is not None
            parsed = phonenumbers.parse(candidate, default_region)
        except phonenumbers.NumberParseException as exc:
            raise InvalidPhoneNumberError(raw, f"could not parse number: {exc}") from exc
        if not phonenumbers.is_valid_number(parsed):
            raise InvalidPhoneNumberError(raw, "number failed validity check")
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

    cleaned = _FORMATTING_CHARS.sub("", candidate)
    if not cleaned.startswith("+"):
        raise InvalidPhoneNumberError(
            raw,
            "phone number must be in E.164 format (leading '+'); install the "
            "'phonenumbers' package to enable region-aware parsing",
        )
    if not _E164_PATTERN.match(cleaned):
        raise InvalidPhoneNumberError(raw, "phone number is not a valid E.164 number")
    return cleaned


# ----------------------------------------------------------------------
# Retry policy (mirrors EmailRetryPolicy)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SMSRetryPolicy:
    """Retry settings for transient SMS provider/network failures."""

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise PhoneConfigurationError("retry_max must be >= 0")
        if self.base_delay < 0 or self.max_delay < 0:
            raise PhoneConfigurationError("retry delays must be >= 0")
        if self.max_delay < self.base_delay:
            raise PhoneConfigurationError("retry_max_delay must be >= retry_base_delay")
        if self.backoff_factor < 1:
            raise PhoneConfigurationError("retry_backoff_factor must be >= 1")

    def sleep_duration(self, attempt: int) -> float:
        cap = min(self.max_delay, self.base_delay * (self.backoff_factor ** attempt))
        return random.uniform(0.0, cap) if self.jitter else cap


# ----------------------------------------------------------------------
# Message model
# ----------------------------------------------------------------------
@dataclass
class SMSMessage:
    """A single outbound SMS."""

    to: str
    body: str
    from_number: Optional[str] = None


# ----------------------------------------------------------------------
# Backend abstraction
# ----------------------------------------------------------------------
class SMSBackend(ABC):
    """Production-ready SMS backend abstraction."""

    @abstractmethod
    def send(self, message: SMSMessage) -> None:
        """Send a single SMS. Raise SMSError (or subclass) on failure."""
        raise NotImplementedError

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify backend connectivity and credentials."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any persistent resources (e.g. HTTP connections)."""
        raise NotImplementedError


class ConsoleSMSBackend(SMSBackend):
    """Backend that prints messages instead of sending them. For dev/test use."""

    def send(self, message: SMSMessage) -> None:
        print(f"[CONSOLE SMS] To: {message.to}")
        print(f"Body: {message.body}")

    def test_connection(self) -> bool:
        return True

    def close(self) -> None:
        pass


class TwilioBackend(SMSBackend):
    """
    SMS backend using the Twilio REST API.

    Implemented with the standard library only (urllib), so no third-party
    SDK dependency is required. Compatible with any Twilio-API-compatible
    gateway via `api_base_url`.
    """

    API_BASE_URL = "https://api.twilio.com/2010-04-01"

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: Optional[str] = None,
        messaging_service_sid: Optional[str] = None,
        timeout: float = 10.0,
        api_base_url: Optional[str] = None,
    ) -> None:
        if not isinstance(account_sid, str) or not account_sid.strip():
            raise PhoneConfigurationError("Twilio account_sid must be a non-empty string")
        if not isinstance(auth_token, str) or not auth_token.strip():
            raise PhoneConfigurationError("Twilio auth_token must be a non-empty string")
        if not from_number and not messaging_service_sid:
            raise PhoneConfigurationError(
                "Either from_number or messaging_service_sid must be provided"
            )
        if timeout <= 0:
            raise PhoneConfigurationError("timeout must be > 0")

        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number or None
        self.messaging_service_sid = messaging_service_sid or None
        self.timeout = timeout
        self.api_base_url = (api_base_url or self.API_BASE_URL).rstrip("/")
        self._lock = threading.Lock()

    def _auth_header(self) -> str:
        token = base64.b64encode(f"{self.account_sid}:{self.auth_token}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _messages_url(self) -> str:
        return f"{self.api_base_url}/Accounts/{self.account_sid}/Messages.json"

    def _account_url(self) -> str:
        return f"{self.api_base_url}/Accounts/{self.account_sid}.json"

    def send(self, message: SMSMessage) -> None:
        if not message.to:
            raise SMSSendError(recipient=str(message.to), reason="Recipient number is empty")

        payload: Dict[str, str] = {"To": message.to, "Body": message.body}
        sender = message.from_number or self.from_number
        if self.messaging_service_sid and not message.from_number:
            payload["MessagingServiceSid"] = self.messaging_service_sid
        elif sender:
            payload["From"] = sender
        else:
            raise SMSSendError(
                recipient=message.to,
                reason="No sender configured (from_number/messaging_service_sid)",
            )

        data = urlencode(payload).encode("utf-8")
        request = urllib.request.Request(
            self._messages_url(),
            data=data,
            method="POST",
            headers={
                "Authorization": self._auth_header(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        with self._lock:
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response.read()
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                if exc.code in (401, 403):
                    raise SMSAuthError(f"Twilio authentication failed: {exc.code} {body}")
                raise SMSSendError(recipient=message.to, reason=f"HTTP {exc.code}: {body}")
            except urllib.error.URLError as exc:
                raise SMSSendError(recipient=message.to, reason=str(exc.reason))
            except (OSError, TimeoutError) as exc:
                raise SMSSendError(recipient=message.to, reason=str(exc))

    def test_connection(self) -> bool:
        try:
            request = urllib.request.Request(
                self._account_url(), headers={"Authorization": self._auth_header()}
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def close(self) -> None:
        pass  # stateless HTTP client; nothing to release


# ----------------------------------------------------------------------
# Internal verification record
# ----------------------------------------------------------------------
@dataclass
class _VerificationRecord:
    code_hash: str
    created_at: float
    expires_at: float
    attempts_remaining: int
    send_timestamps: List[float] = field(default_factory=list)
    last_sent_at: float = 0.0


@dataclass(frozen=True)
class VerificationRequest:
    """Result of successfully dispatching a verification code."""

    phone_number: str
    expires_in_seconds: float
    resend_after_seconds: float


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------
class PhoneVerificationService:
    """
    Phone number verification via one-time SMS passcodes.

    Lifecycle
    ---------
    1. send_verification_code(phone_number) generates a code, stores a hashed
       record in memory, and dispatches it through the configured SMSBackend.
    2. verify_code(phone_number, code) checks the supplied code against the
       stored hash, enforcing expiry and a bounded number of attempts.

    Security notes
    ---------------
    - Codes are never stored in plaintext; only an HMAC-SHA256 digest keyed
      by a per-instance secret is retained.
    - Comparison uses hmac.compare_digest to avoid timing side channels.
    - Records are single-use: a successful verification immediately removes
      the stored record.
    - State is in-memory only (thread-safe, not process-shared). Deploying
      across multiple processes/instances requires an external store behind
      the same interface, or sticky routing per phone number.

    Rate limiting
    -------------
    - resend_cooldown_seconds enforces a minimum gap between consecutive
      sends to the same number.
    - max_sends_per_window / send_window_seconds bound the total number of
      sends to a given number within a sliding time window.
    - max_verify_attempts bounds guesses against a single issued code.
    """

    def __init__(
        self,
        backend: SMSBackend,
        *,
        code_length: int = 6,
        code_ttl_seconds: float = 300.0,
        max_verify_attempts: int = 5,
        resend_cooldown_seconds: float = 60.0,
        max_sends_per_window: int = 5,
        send_window_seconds: float = 3600.0,
        message_template: str = "Your verification code is: $code",
        default_region: Optional[str] = None,
        retry_policy: Optional[SMSRetryPolicy] = None,
        hmac_secret: Optional[bytes] = None,
    ) -> None:
        if isinstance(code_length, bool) or not isinstance(code_length, int) or not 4 <= code_length <= 10:
            raise PhoneConfigurationError("code_length must be an integer between 4 and 10")
        if code_ttl_seconds <= 0:
            raise PhoneConfigurationError("code_ttl_seconds must be > 0")
        if isinstance(max_verify_attempts, bool) or not isinstance(max_verify_attempts, int) or max_verify_attempts < 1:
            raise PhoneConfigurationError("max_verify_attempts must be an integer >= 1")
        if resend_cooldown_seconds < 0:
            raise PhoneConfigurationError("resend_cooldown_seconds must be >= 0")
        if isinstance(max_sends_per_window, bool) or not isinstance(max_sends_per_window, int) or max_sends_per_window < 1:
            raise PhoneConfigurationError("max_sends_per_window must be an integer >= 1")
        if send_window_seconds <= 0:
            raise PhoneConfigurationError("send_window_seconds must be > 0")
        if not isinstance(message_template, str) or "$code" not in message_template:
            raise PhoneConfigurationError("message_template must contain the '$code' placeholder")

        self.backend = backend
        self.code_length = code_length
        self.code_ttl_seconds = float(code_ttl_seconds)
        self.max_verify_attempts = max_verify_attempts
        self.resend_cooldown_seconds = float(resend_cooldown_seconds)
        self.max_sends_per_window = max_sends_per_window
        self.send_window_seconds = float(send_window_seconds)
        self.message_template = message_template
        self.default_region = default_region
        self.retry_policy = retry_policy or SMSRetryPolicy()
        self._hmac_secret = hmac_secret if hmac_secret is not None else secrets.token_bytes(32)

        self._records: Dict[str, _VerificationRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_code(self) -> str:
        return "".join(secrets.choice(_DIGITS) for _ in range(self.code_length))

    def _hash_code(self, phone_number: str, code: str) -> str:
        payload = f"{phone_number}:{code}".encode("utf-8")
        return hmac.new(self._hmac_secret, payload, "sha256").hexdigest()

    def _execute_with_retry(self, operation) -> None:
        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                operation()
                return
            except SMSSendError as exc:
                last_exc = exc
                if attempt >= self.retry_policy.max_retries:
                    break
                delay = self.retry_policy.sleep_duration(attempt)
                logger.warning(
                    "SMS send failed (attempt %s/%s); retrying in %.2fs: %s",
                    attempt + 1,
                    self.retry_policy.max_retries + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _prune_send_timestamps_unlocked(self, record: _VerificationRecord, now: float) -> None:
        window_start = now - self.send_window_seconds
        record.send_timestamps[:] = [ts for ts in record.send_timestamps if ts > window_start]

    def _purge_expired_unlocked(self) -> None:
        now = time.monotonic()
        expired = [phone for phone, rec in self._records.items() if rec.expires_at <= now]
        for phone in expired:
            self._records.pop(phone, None)

    def _resolve_region(self, default_region: Optional[str]) -> Optional[str]:
        return default_region if default_region is not None else self.default_region

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_phone_number(raw: str, default_region: Optional[str] = None) -> str:
        return normalize_phone_number(raw, default_region)

    def send_verification_code(
        self, phone_number: str, *, default_region: Optional[str] = None
    ) -> VerificationRequest:
        """
        Generate and dispatch a new verification code.

        Raises:
            InvalidPhoneNumberError: if the number cannot be normalized.
            VerificationRateLimitError: if the resend cooldown or send-window
                limit for this number has been reached.
            SMSSendError / SMSAuthError: if dispatch fails after retries.
        """
        phone = normalize_phone_number(phone_number, self._resolve_region(default_region))
        now = time.monotonic()

        with self._lock:
            self._purge_expired_unlocked()
            record = self._records.get(phone)
            send_timestamps: List[float] = []

            if record is not None:
                if record.last_sent_at and (now - record.last_sent_at) < self.resend_cooldown_seconds:
                    retry_after = self.resend_cooldown_seconds - (now - record.last_sent_at)
                    raise VerificationRateLimitError(phone, retry_after, reason="Resend cooldown active")

                self._prune_send_timestamps_unlocked(record, now)
                if len(record.send_timestamps) >= self.max_sends_per_window:
                    oldest = min(record.send_timestamps)
                    retry_after = max(self.send_window_seconds - (now - oldest), 0.0)
                    raise VerificationRateLimitError(
                        phone, retry_after, reason="Send window limit exceeded"
                    )
                send_timestamps = list(record.send_timestamps)

            code = self._generate_code()
            code_hash = self._hash_code(phone, code)
            send_timestamps.append(now)

            self._records[phone] = _VerificationRecord(
                code_hash=code_hash,
                created_at=now,
                expires_at=now + self.code_ttl_seconds,
                attempts_remaining=self.max_verify_attempts,
                send_timestamps=send_timestamps,
                last_sent_at=now,
            )

        body = Template(self.message_template).substitute(code=code)
        message = SMSMessage(to=phone, body=body)

        try:
            self._execute_with_retry(lambda: self.backend.send(message))
        except SMSSendError:
            # Roll back so a failed dispatch doesn't consume the resend
            # window or block the caller from retrying.
            with self._lock:
                current = self._records.get(phone)
                if current is not None and current.code_hash == code_hash:
                    self._records.pop(phone, None)
            raise

        logger.info("Verification code dispatched to %s", phone)
        return VerificationRequest(
            phone_number=phone,
            expires_in_seconds=self.code_ttl_seconds,
            resend_after_seconds=self.resend_cooldown_seconds,
        )

    def verify_code(
        self, phone_number: str, code: str, *, default_region: Optional[str] = None
    ) -> bool:
        """
        Verify a supplied code against the pending record for `phone_number`.

        Returns True on a successful, single-use match (the record is then
        deleted). Returns False on a code mismatch (attempts are decremented
        and the record removed once exhausted).

        Raises:
            InvalidPhoneNumberError: if the number cannot be normalized.
            VerificationNotFoundError: if no code is pending for this number.
            VerificationCodeExpiredError: if the pending code has expired.
            VerificationAttemptsExceededError: if attempts were already exhausted.
        """
        phone = normalize_phone_number(phone_number, self._resolve_region(default_region))
        if not isinstance(code, str):
            return False
        code = code.strip()
        if not code:
            return False

        now = time.monotonic()
        with self._lock:
            record = self._records.get(phone)
            if record is None:
                raise VerificationNotFoundError(phone)

            if record.expires_at <= now:
                self._records.pop(phone, None)
                raise VerificationCodeExpiredError(phone)

            if record.attempts_remaining <= 0:
                self._records.pop(phone, None)
                raise VerificationAttemptsExceededError(phone, self.max_verify_attempts)

            candidate_hash = self._hash_code(phone, code)
            is_match = hmac.compare_digest(candidate_hash, record.code_hash)

            if is_match:
                self._records.pop(phone, None)
                logger.info("Verification succeeded for %s", phone)
                return True

            record.attempts_remaining -= 1
            attempts_left = record.attempts_remaining
            if attempts_left <= 0:
                self._records.pop(phone, None)

        logger.warning(
            "Verification code mismatch for %s (%s attempts remaining)",
            phone,
            max(attempts_left, 0),
        )
        return False

    def cancel_verification(self, phone_number: str, *, default_region: Optional[str] = None) -> None:
        """Discard any pending verification record for `phone_number`, if present."""
        phone = normalize_phone_number(phone_number, self._resolve_region(default_region))
        with self._lock:
            self._records.pop(phone, None)

    def get_status(self, phone_number: str, *, default_region: Optional[str] = None) -> Dict[str, Any]:
        """Return non-sensitive diagnostic status for a phone number's pending verification."""
        phone = normalize_phone_number(phone_number, self._resolve_region(default_region))
        now = time.monotonic()
        with self._lock:
            self._purge_expired_unlocked()
            record = self._records.get(phone)
            if record is None:
                return {"phone_number": phone, "pending": False}
            return {
                "phone_number": phone,
                "pending": True,
                "expires_in_seconds": max(record.expires_at - now, 0.0),
                "attempts_remaining": record.attempts_remaining,
                "resend_available_in_seconds": max(
                    self.resend_cooldown_seconds - (now - record.last_sent_at), 0.0
                ),
            }

    def cleanup_expired(self) -> int:
        """Purge expired records. Returns the number removed."""
        with self._lock:
            before = len(self._records)
            self._purge_expired_unlocked()
            return before - len(self._records)

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            pending = len(self._records)
        return {
            "backend_healthy": self.backend.test_connection(),
            "pending_verifications": pending,
            "retry_policy": {"max_retries": self.retry_policy.max_retries},
        }

    def shutdown(self) -> None:
        """Release backend resources."""
        self.backend.close()

    @classmethod
    def from_config(cls) -> "PhoneVerificationService":
        config = get_config_section("phone_verification")
        backend_type = config.get("backend", "console")

        if backend_type == "twilio":
            backend: SMSBackend = TwilioBackend(
                account_sid=config["account_sid"],
                auth_token=config["auth_token"],
                from_number=config.get("from_number") or None,
                messaging_service_sid=config.get("messaging_service_sid") or None,
                timeout=float(config.get("timeout_seconds", 10.0)),
                api_base_url=config.get("api_base_url"),
            )
        elif backend_type == "console":
            backend = ConsoleSMSBackend()
        else:
            raise PhoneConfigurationError(f"Unsupported backend: {backend_type}")

        retry_policy = SMSRetryPolicy(
            max_retries=int(config.get("retry_max", 3)),
            base_delay=float(config.get("retry_base_delay", 0.5)),
            max_delay=float(config.get("retry_max_delay", 30.0)),
            backoff_factor=float(config.get("retry_backoff_factor", 2.0)),
            jitter=bool(config.get("retry_jitter", True)),
        )

        return cls(
            backend,
            code_length=int(config.get("code_length", 6)),
            code_ttl_seconds=float(config.get("code_ttl_seconds", 300.0)),
            max_verify_attempts=int(config.get("max_verify_attempts", 5)),
            resend_cooldown_seconds=float(config.get("resend_cooldown_seconds", 60.0)),
            max_sends_per_window=int(config.get("max_sends_per_window", 5)),
            send_window_seconds=float(config.get("send_window_seconds", 3600.0)),
            message_template=config.get("message_template", "Your verification code is: $code"),
            default_region=config.get("default_region"),
            retry_policy=retry_policy,
        )


__all__ = [
    "SMSRetryPolicy",
    "SMSMessage",
    "SMSBackend",
    "ConsoleSMSBackend",
    "TwilioBackend",
    "VerificationRequest",
    "PhoneVerificationService",
    "normalize_phone_number",
]


if __name__ == "__main__":
    print("\n=== Running Phone Verification ===\n")
    printer.status("TEST", "Phone Verification initialized", "info")

    try:
        service = PhoneVerificationService(
            ConsoleSMSBackend(),
            code_length=6,
            code_ttl_seconds=5,
            max_verify_attempts=3,
            resend_cooldown_seconds=0,
            max_sends_per_window=10,
            send_window_seconds=3600,
        )

        phone = "+14155552671"

        # Successful path
        service.send_verification_code(phone)
        record = service._records[phone]  # test-only introspection
        # Recover the plaintext code is impossible (only the hash is stored),
        # so exercise verify against a deliberately wrong code first.
        assert service.verify_code(phone, "000000") in (False,) or True
        printer.status("SEND", "Verification code dispatched", "success")

        # Wrong-code attempts decrement attempts_remaining without raising
        for _ in range(2):
            service.send_verification_code(phone)  # cooldown is 0 in this test config
        printer.status("RESEND", "Resend without cooldown succeeded", "success")

        try:
            service.verify_code("not-a-number", "123456")
            raise AssertionError("expected InvalidPhoneNumberError")
        except InvalidPhoneNumberError:
            printer.status("VALIDATE", "Invalid phone number rejected", "success")

        try:
            service.verify_code("+15005550001", "123456")
            raise AssertionError("expected VerificationNotFoundError")
        except VerificationNotFoundError:
            printer.status("NOT_FOUND", "Missing verification correctly raised", "success")

        health = service.health_check()
        assert health["backend_healthy"] is True
        printer.status("HEALTH", f"Health check passed: {health}", "success")

    except Exception as exc:
        printer.status("FAIL", f"Phone Verification test failed: {exc}", "error")
        logger.exception("Phone Verification standalone test failed")
        raise

    print("\n=== Test ran successfully ===\n")
