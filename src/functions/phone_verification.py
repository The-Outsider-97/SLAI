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
import phonenumbers # type: ignore

from phonenumbers import PhoneNumberFormat # type: ignore
from abc import ABC, abstractmethod
from dataclasses import dataclass
from string import Template
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from .utils.config_loader import get_config_section
from .utils.functions_error import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Phone Verification")
printer = PrettyPrinter()


OTP_CODE_LENGTH = 6

_OTP_PATTERN = re.compile(r"^\d{6}$")
_DIGITS = string.digits


# ----------------------------------------------------------------------
# Phone number normalization and country validation
# ----------------------------------------------------------------------
def _normalize_region_code(region: Optional[str]) -> Optional[str]:
    if region is None:
        return None

    if not isinstance(region, str):
        raise InvalidCountryCodeError(
            str(region),
            "country region must be a two-letter ISO 3166-1 region code",
        )

    normalized = region.strip().upper()

    if normalized not in phonenumbers.SUPPORTED_REGIONS:
        raise InvalidCountryCodeError(
            normalized,
            "unsupported ISO 3166-1 phone region",
        )

    return normalized


def _normalize_country_calling_code(country_code: Any) -> int:
    if isinstance(country_code, bool):
        raise InvalidCountryCodeError(
            str(country_code),
            "country calling code must be numeric",
        )

    value = str(country_code).strip()

    if value.startswith("+"):
        value = value[1:]

    if (
        not value
        or not value.isdigit()
        or value.startswith("0")
        or len(value) > 3
    ):
        raise InvalidCountryCodeError(
            str(country_code),
            "country calling code must be in the form '+31', '31', or 31",
        )

    normalized = int(value)

    regions = phonenumbers.region_codes_for_country_code(normalized)

    if not regions or all(region == "001" for region in regions):
        raise InvalidCountryCodeError(
            str(country_code),
            "calling code does not identify a supported geographic region",
        )

    return normalized


def _parse_valid_phone_number(raw: str, default_region: Optional[str] = None,):
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidPhoneNumberError(
            str(raw),
            "phone number must be a non-empty string",
        )

    region = _normalize_region_code(default_region)
    candidate = raw.strip()

    if not candidate.startswith("+") and region is None:
        raise InvalidPhoneNumberError(
            raw,
            "national-format phone numbers require a country region",
        )

    try:
        parsed = phonenumbers.parse(candidate, region)
    except phonenumbers.NumberParseException as exc:
        raise InvalidPhoneNumberError(
            raw,
            f"could not parse number: {exc}",
        ) from exc

    if not phonenumbers.is_possible_number(parsed):
        raise InvalidPhoneNumberError(
            raw,
            "phone number is not possible for its numbering plan",
        )

    if not phonenumbers.is_valid_number(parsed):
        raise InvalidPhoneNumberError(
            raw,
            "phone number failed numbering-plan validation",
        )

    return parsed


def normalize_phone_number(raw: str, default_region: Optional[str] = None,) -> str:
    """
    Normalize a valid phone number to E.164.

    Example:
        06 12345678 + region NL -> +31612345678
    """
    parsed = _parse_valid_phone_number(raw, default_region)

    return phonenumbers.format_number(
        parsed,
        PhoneNumberFormat.E164,
    )


def validate_phone_country(raw: str, country_code: Any, country_region: Optional[str] = None,) -> str:
    """
    Validate and normalize a phone number against the selected country.

    `country_code` is the international calling code:
        +31, 31, or 31

    `country_region` is the ISO 3166-1 alpha-2 region:
        NL, US, CW, BQ, GB, etc.

    When `country_region` is supplied, both the calling code and the
    region-specific numbering plan must match.

    Returns:
        Canonical E.164 phone number.

    Raises:
        InvalidPhoneNumberError
        InvalidCountryCodeError
        PhoneCountryMismatchError
    """
    expected_calling_code = _normalize_country_calling_code(country_code)
    expected_region = _normalize_region_code(country_region)

    if expected_region is not None:
        region_calling_code = phonenumbers.country_code_for_region(
            expected_region
        )

        if region_calling_code != expected_calling_code:
            raise InvalidCountryCodeError(
                f"+{expected_calling_code}",
                (
                    f"calling code +{expected_calling_code} does not match "
                    f"region {expected_region}, which uses "
                    f"+{region_calling_code}"
                ),
            )

    parsed = _parse_valid_phone_number(raw, expected_region,)

    actual_calling_code = int(parsed.country_code)
    actual_region = phonenumbers.region_code_for_number(parsed)

    if actual_calling_code != expected_calling_code:
        raise PhoneCountryMismatchError(
            raw,
            expected_country_code=f"+{expected_calling_code}",
            actual_country_code=f"+{actual_calling_code}",
            expected_region=expected_region,
            actual_region=actual_region,
        )

    if (
        expected_region is not None
        and not phonenumbers.is_valid_number_for_region(
            parsed,
            expected_region,
        )
    ):
        raise PhoneCountryMismatchError(
            raw,
            expected_country_code=f"+{expected_calling_code}",
            actual_country_code=f"+{actual_calling_code}",
            expected_region=expected_region,
            actual_region=actual_region,
        )

    return phonenumbers.format_number(
        parsed,
        PhoneNumberFormat.E164,
    )


def _mask_phone_number(phone_number: str) -> str:
    """Return a log-safe representation of an E.164 phone number."""
    if len(phone_number) <= 6:
        return "***"

    return f"{phone_number[:3]}***{phone_number[-4:]}"

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


@dataclass(frozen=True)
class VerificationRequest:
    """Result of successfully dispatching a verification code."""

    phone_number: str
    country_code: str
    country_region: Optional[str]
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
        if (
            isinstance(code_length, bool)
            or not isinstance(code_length, int)
            or code_length != OTP_CODE_LENGTH
        ):
            raise PhoneConfigurationError(
                f"code_length must be exactly {OTP_CODE_LENGTH}"
            )
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
        self._send_history: Dict[str, List[float]] = {}
        self._pending_sends: set[str] = set()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_code(self) -> str:
        return "".join(secrets.choice(_DIGITS) for _ in range(OTP_CODE_LENGTH))

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

    def _prune_send_history_unlocked(self, phone_number: str, now: float,) -> List[float]:
        window_start = now - self.send_window_seconds

        history = [
            timestamp
            for timestamp in self._send_history.get(phone_number, [])
            if timestamp > window_start
        ]

        if history:
            self._send_history[phone_number] = history
        else:
            self._send_history.pop(phone_number, None)

        return history

    def _purge_expired_unlocked(self) -> None:
        now = time.monotonic()

        expired = [
            phone
            for phone, record in self._records.items()
            if record.expires_at <= now
        ]

        for phone in expired:
            self._records.pop(phone, None)

        for phone in tuple(self._send_history):
            self._prune_send_history_unlocked(phone, now)

    def _resolve_region(self, default_region: Optional[str]) -> Optional[str]:
        return default_region if default_region is not None else self.default_region

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_phone_number(raw: str, default_region: Optional[str] = None) -> str:
        return normalize_phone_number(raw, default_region)

    def send_verification_code(self, phone_number: str, *, country_code: Any, country_region: Optional[str] = None) -> VerificationRequest:
        """
        Validate the phone number against the selected country and,
        only after successful validation, dispatch a six-digit OTP.

        Args:
            phone_number:
                User-supplied phone number.

            country_code:
                Selected international calling code, e.g. "+31".

            country_region:
                Optional ISO 3166-1 alpha-2 region, e.g. "NL".
                Strongly recommended because some countries share a
                calling code.

        Raises:
            InvalidPhoneNumberError
            InvalidCountryCodeError
            PhoneCountryMismatchError
            VerificationRateLimitError
            SMSSendError
            SMSAuthError
        """
        region = (
            country_region
            if country_region is not None
            else self.default_region
        )

        phone = validate_phone_country(phone_number, country_code, region,)
        expected_country_code = (f"+{_normalize_country_calling_code(country_code)}")
        normalized_region = _normalize_region_code(region)

        now = time.monotonic()

        with self._lock:
            self._purge_expired_unlocked()

            if phone in self._pending_sends:
                raise VerificationRateLimitError(phone, 1.0, reason="Verification SMS dispatch already in progress")
            send_history = self._prune_send_history_unlocked(phone, now,)
            if send_history:
                elapsed = now - send_history[-1]

                if elapsed < self.resend_cooldown_seconds:
                    retry_after = (
                        self.resend_cooldown_seconds - elapsed
                    )

                    raise VerificationRateLimitError(phone, retry_after, reason="Resend cooldown active")

            if len(send_history) >= self.max_sends_per_window:
                oldest = send_history[0]

                retry_after = max(self.send_window_seconds - (now - oldest), 0.0)

                raise VerificationRateLimitError(phone, retry_after, reason="Send window limit exceeded")

            self._pending_sends.add(phone)

        code = self._generate_code()
        code_hash = self._hash_code(phone, code)
        body = Template(self.message_template).substitute(code=code)
        message = SMSMessage(to=phone, body=body)

        try:
            self._execute_with_retry(
                lambda: self.backend.send(message)
            )

            sent_at = time.monotonic()

            with self._lock:
                history = self._prune_send_history_unlocked(phone, sent_at)
                history.append(sent_at)
                self._send_history[phone] = history

                self._records[phone] = _VerificationRecord(
                    code_hash=code_hash,
                    created_at=sent_at,
                    expires_at=sent_at + self.code_ttl_seconds,
                    attempts_remaining=self.max_verify_attempts,
                )

        finally:
            with self._lock:
                self._pending_sends.discard(phone)

        logger.info("Verification code dispatched to %s", _mask_phone_number(phone))

        return VerificationRequest(
            phone_number=phone,
            country_code=expected_country_code,
            country_region=normalized_region,
            expires_in_seconds=self.code_ttl_seconds,
            resend_after_seconds=self.resend_cooldown_seconds,
        )

    def verify_code(self, phone_number: str, code: str, *, default_region: Optional[str] = None) -> bool:
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
        
        if not isinstance(code, str):
            return False

        code = code.strip()

        if not _OTP_PATTERN.fullmatch(code):
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
                logger.info("Verification succeeded for %s", _mask_phone_number(phone))
                return True

            record.attempts_remaining -= 1
            attempts_left = record.attempts_remaining
            if attempts_left <= 0:
                self._records.pop(phone, None)

        logger.warning(
            "Verification code mismatch for %s (%s attempts remaining)",
            _mask_phone_number(phone),
            max(attempts_left, 0),
        )
        return False

    def cancel_verification(self, phone_number: str, *, default_region: Optional[str] = None) -> None:
        """Discard any pending verification record for `phone_number`, if present."""
        phone = normalize_phone_number(phone_number, self._resolve_region(default_region))
        with self._lock:
            self._records.pop(phone, None)

    def get_status(self, phone_number: str, *, default_region: Optional[str] = None) -> Dict[str, Any]:
        """Return non-sensitive status for a pending verification."""

        phone = normalize_phone_number(phone_number, self._resolve_region(default_region))
        now = time.monotonic()

        with self._lock:
            self._purge_expired_unlocked()

            record = self._records.get(phone)
            history = self._prune_send_history_unlocked(phone, now)
            last_sent_at = history[-1] if history else 0.0

            resend_available = 0.0
            if last_sent_at:
                resend_available = max(
                    self.resend_cooldown_seconds
                    - (now - last_sent_at),
                    0.0,
                )

            if record is None:
                return {
                    "phone_number": phone,
                    "pending": False,
                    "dispatch_in_progress": (
                        phone in self._pending_sends
                    ),
                    "resend_available_in_seconds": resend_available,
                }

            return {
                "phone_number": phone,
                "pending": True,
                "dispatch_in_progress": (
                    phone in self._pending_sends
                ),
                "expires_in_seconds": max(
                    record.expires_at - now,
                    0.0,
                ),
                "attempts_remaining": record.attempts_remaining,
                "resend_available_in_seconds": resend_available,
            }

    def cleanup_expired(self) -> int:
        """Purge expired records. Returns the number removed."""
        with self._lock:
            before = len(self._records)
            self._purge_expired_unlocked()
            return before - len(self._records)

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            self._purge_expired_unlocked()

            pending = len(self._records)
            dispatching = len(self._pending_sends)

        return {
            "backend_healthy": self.backend.test_connection(),
            "pending_verifications": pending,
            "sms_dispatches_in_progress": dispatching,
            "otp_code_length": OTP_CODE_LENGTH,
            "retry_policy": {
                "max_retries": self.retry_policy.max_retries,
            },
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
    "OTP_CODE_LENGTH",
    "SMSRetryPolicy",
    "SMSMessage",
    "SMSBackend",
    "ConsoleSMSBackend",
    "TwilioBackend",
    "VerificationRequest",
    "PhoneVerificationService",
    "normalize_phone_number",
    "validate_phone_country",
]


if __name__ == "__main__":
    print("\n=== Running Phone Verification ===\n")
    printer.status("TEST", "Phone Verification initialized", "info")

    class _CaptureSMSBackend(SMSBackend):
        def __init__(self) -> None:
            self.messages: List[SMSMessage] = []

        def send(self, message: SMSMessage) -> None:
            self.messages.append(message)

        def test_connection(self) -> bool:
            return True

        def close(self) -> None:
            pass

    try:
        backend = _CaptureSMSBackend()

        service = PhoneVerificationService(
            backend,
            code_length=6,
            code_ttl_seconds=300,
            max_verify_attempts=3,
            resend_cooldown_seconds=0,
            max_sends_per_window=10,
            send_window_seconds=3600,
        )

        # --------------------------------------------------------------
        # Obtain a metadata-defined Dutch example number instead of
        # using a real user's phone number.
        # --------------------------------------------------------------
        example = phonenumbers.example_number_for_type("NL", phonenumbers.PhoneNumberType.MOBILE)
        if example is None:
            example = phonenumbers.example_number("NL")

        assert example is not None
        phone = phonenumbers.format_number(example, PhoneNumberFormat.E164)

        # --------------------------------------------------------------
        # Correct country/region
        # --------------------------------------------------------------
        request = service.send_verification_code(
            phone,
            country_code="+31",
            country_region="NL",
        )

        assert request.phone_number == phone
        assert request.country_code == "+31"
        assert request.country_region == "NL"

        assert len(backend.messages) == 1

        match = re.search(
            r"\b(\d{6})\b",
            backend.messages[-1].body,
        )

        assert match is not None

        otp = match.group(1)

        assert len(otp) == 6
        assert otp.isdigit()

        printer.status("SEND", "Six-digit verification code dispatched", "success")

        # --------------------------------------------------------------
        # Successful OTP verification
        # --------------------------------------------------------------
        assert service.verify_code(
            phone,
            otp,
        ) is True

        printer.status("VERIFY", "Verification code accepted", "success")

        # --------------------------------------------------------------
        # Country mismatch must fail BEFORE SMS dispatch
        # --------------------------------------------------------------
        sent_before = len(backend.messages)

        try:
            service.send_verification_code(
                phone,
                country_code="+44",
                country_region="GB",
            )
            raise AssertionError(
                "expected PhoneCountryMismatchError"
            )
        except PhoneCountryMismatchError:
            pass

        assert len(backend.messages) == sent_before

        printer.status("COUNTRY", "Country mismatch rejected before SMS dispatch", "success")

        # --------------------------------------------------------------
        # Invalid calling-code/region combination
        # --------------------------------------------------------------
        try:
            service.send_verification_code(
                phone,
                country_code="+1",
                country_region="NL",
            )
            raise AssertionError(
                "expected InvalidCountryCodeError"
            )
        except InvalidCountryCodeError:
            pass

        printer.status("COUNTRY_CODE", "Invalid country-code/region pair rejected", "success")

        # --------------------------------------------------------------
        # Invalid phone number
        # --------------------------------------------------------------
        try:
            service.send_verification_code(
                "not-a-number",
                country_code="+31",
                country_region="NL",
            )
            raise AssertionError("expected InvalidPhoneNumberError")
        except InvalidPhoneNumberError:
            pass

        printer.status("VALIDATE",
            "Invalid phone number rejected",
            "success",
        )

        health = service.health_check()

        assert health["backend_healthy"] is True
        assert health["otp_code_length"] == 6

        printer.status(
            "HEALTH",
            f"Health check passed: {health}",
            "success",
        )

        service.shutdown()

    except Exception as exc:
        printer.status("FAIL", f"Phone Verification test failed: {exc}", "error")
        logger.exception("Phone Verification standalone test failed")
        raise

    print("\n=== Test ran successfully ===\n")