"""Provider-neutral payment orchestration for SLAI application integrations.

Design goals
------------
* Keep payment state and validation in SLAI while leaving gateway-specific
  SDK/API calls behind a small ``PaymentProvider`` protocol.
* Never accept or persist raw PAN/CVV data. Callers must supply a provider
  token/reference instead.
* Use ``Decimal`` for monetary values.
* Preserve idempotency for authorize/capture/refund operations.
* Allow persistence and idempotency backends to be replaced without changing
  application call sites.

This module deliberately does not embed Stripe, Adyen, PayPal, or another
gateway SDK. A production adapter should implement ``PaymentProvider`` and
translate its provider's responses/exceptions into this contract.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Union, runtime_checkable

from .utils.functions_error import *
from .utils.functions_helpers import *
from logs.logger import get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Payment")

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:-]{8,200}$")
_RAW_CARD_RE = re.compile(r"^\d{13,19}$")
_MAX_REFERENCE_LENGTH = 200
_MAX_DESCRIPTION_LENGTH = 500
_MAX_TOKEN_LENGTH = 1024

DecimalLike = Union[str, int, Decimal]


# -------------------------------------------------------------------------
# Domain types
# -------------------------------------------------------------------------
class PaymentStatus(str, Enum):
    """Local payment lifecycle states."""

    AUTHORIZED = "authorized"
    PARTIALLY_CAPTURED = "partially_captured"
    CAPTURED = "captured"
    PARTIALLY_REFUNDED = "partially_refunded"
    REFUNDED = "refunded"
    VOIDED = "voided"
    FAILED = "failed"


@dataclass(frozen=True)
class Money:
    """Currency-qualified monetary amount using exact decimal arithmetic."""

    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        amount = _coerce_decimal(self.amount, "amount")
        currency = str(self.currency).strip().upper()

        if not amount.is_finite():
            raise PaymentValidationError("amount must be finite")
        if amount <= 0:
            raise PaymentValidationError("amount must be greater than zero")
        if not _CURRENCY_RE.fullmatch(currency):
            raise PaymentValidationError(
                "currency must be a three-letter ISO-style code such as EUR or USD"
            )

        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "currency", currency)

    @classmethod
    def of(cls, amount: DecimalLike, currency: str) -> "Money":
        return cls(amount=_coerce_decimal(amount, "amount"), currency=currency)

    def as_dict(self) -> Dict[str, str]:
        return {"amount": format(self.amount, "f"), "currency": self.currency}


@dataclass(frozen=True)
class PaymentRequest:
    """Input for a payment authorization.

    ``payment_method_token`` must be an opaque token/reference created by the
    payment provider or a PCI-compliant tokenization layer. Raw card numbers
    are intentionally rejected.
    """

    money: Money
    payment_method_token: str
    merchant_reference: str
    description: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        token = _require_text(
            self.payment_method_token,
            "payment_method_token",
            max_length=_MAX_TOKEN_LENGTH,
        )
        compact = re.sub(r"[\s-]", "", token)
        if _RAW_CARD_RE.fullmatch(compact):
            raise PaymentValidationError(
                "raw payment-card numbers are not accepted; provide a provider token"
            )

        reference = _require_text(
            self.merchant_reference,
            "merchant_reference",
            max_length=_MAX_REFERENCE_LENGTH,
        )

        description = self.description
        if description is not None:
            description = _require_text(
                description,
                "description",
                max_length=_MAX_DESCRIPTION_LENGTH,
            )

        metadata = _normalize_metadata(self.metadata)

        object.__setattr__(self, "payment_method_token", token)
        object.__setattr__(self, "merchant_reference", reference)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class AuthorizationResult:
    """Result returned by a provider after authorization."""

    provider_payment_id: str
    provider_reference: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_payment_id",
            _require_text(
                self.provider_payment_id,
                "provider_payment_id",
                max_length=500,
            ),
        )
        if self.provider_reference is not None:
            object.__setattr__(
                self,
                "provider_reference",
                _require_text(
                    self.provider_reference,
                    "provider_reference",
                    max_length=500,
                ),
            )
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))


@dataclass(frozen=True)
class ProviderOperationResult:
    """Result for provider capture/refund operations."""

    operation_id: str
    provider_reference: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _require_text(self.operation_id, "operation_id", max_length=500))
        if self.provider_reference is not None:
            object.__setattr__(self, "provider_reference", _require_text(self.provider_reference, "provider_reference", max_length=500))
            object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))


@dataclass
class PaymentRecord:
    """Persistable local representation of a provider-backed payment."""

    payment_id: str
    provider_payment_id: str
    merchant_reference: str
    currency: str
    authorized_amount: Decimal
    captured_amount: Decimal = Decimal("0")
    refunded_amount: Decimal = Decimal("0")
    status: PaymentStatus = PaymentStatus.AUTHORIZED
    provider_reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def remaining_capturable(self) -> Decimal:
        return max(Decimal("0"), self.authorized_amount - self.captured_amount)

    @property
    def remaining_refundable(self) -> Decimal:
        return max(Decimal("0"), self.captured_amount - self.refunded_amount)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "payment_id": self.payment_id,
            "provider_payment_id": self.provider_payment_id,
            "merchant_reference": self.merchant_reference,
            "currency": self.currency,
            "authorized_amount": format(self.authorized_amount, "f"),
            "captured_amount": format(self.captured_amount, "f"),
            "refunded_amount": format(self.refunded_amount, "f"),
            "status": self.status.value,
            "provider_reference": self.provider_reference,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class IdempotencyEntry:
    fingerprint: str
    payment_id: str


# -------------------------------------------------------------------------
# Extension contracts
# -------------------------------------------------------------------------
@runtime_checkable
class PaymentProvider(Protocol):
    """Gateway adapter contract.

    Implementations should use the provider's own idempotency mechanism when
    available and must not log full payment-method tokens.
    """

    def authorize(self, request: PaymentRequest, *, idempotency_key: str) -> AuthorizationResult:
        ...

    def capture(self, provider_payment_id: str, money: Money, *, idempotency_key: str) -> ProviderOperationResult:
        ...

    def refund(self, provider_payment_id: str, money: Money, *, idempotency_key: str) -> ProviderOperationResult:
        ...

    def void(self, provider_payment_id: str, *, idempotency_key: str) -> ProviderOperationResult:
        ...


@runtime_checkable
class PaymentRepository(Protocol):
    """Persistence contract for payment records."""

    def get(self, payment_id: str) -> Optional[PaymentRecord]:
        ...

    def save(self, record: PaymentRecord) -> None:
        ...


@runtime_checkable
class IdempotencyStore(Protocol):
    """Persistence contract for request idempotency."""

    def get(self, namespace: str, key: str) -> Optional[IdempotencyEntry]:
        ...

    def put(self, namespace: str, key: str, entry: IdempotencyEntry) -> None:
        ...


class InMemoryPaymentRepository:
    """Thread-safe repository for development, tests, and single-process use."""

    def __init__(self) -> None:
        self._records: Dict[str, PaymentRecord] = {}
        self._lock = threading.RLock()

    def get(self, payment_id: str) -> Optional[PaymentRecord]:
        with self._lock:
            record = self._records.get(payment_id)
            return _clone_record(record) if record else None

    def save(self, record: PaymentRecord) -> None:
        if not isinstance(record, PaymentRecord):
            raise TypeError("record must be a PaymentRecord")
        with self._lock:
            self._records[record.payment_id] = _clone_record(record)


class InMemoryIdempotencyStore:
    """Thread-safe in-memory idempotency store.

    Distributed deployments should replace this with a shared durable store.
    """

    def __init__(self) -> None:
        self._entries: Dict[tuple[str, str], IdempotencyEntry] = {}
        self._lock = threading.RLock()

    def get(self, namespace: str, key: str) -> Optional[IdempotencyEntry]:
        with self._lock:
            return self._entries.get((namespace, key))

    def put(
        self,
        namespace: str,
        key: str,
        entry: IdempotencyEntry,
    ) -> None:
        with self._lock:
            self._entries[(namespace, key)] = entry


# -------------------------------------------------------------------------
# Service
# -------------------------------------------------------------------------
class PaymentService:
    """Coordinates validated, idempotent payment operations."""

    def __init__(
        self,
        provider: PaymentProvider,
        *,
        repository: Optional[PaymentRepository] = None,
        idempotency_store: Optional[IdempotencyStore] = None,
    ) -> None:
        if not isinstance(provider, PaymentProvider):
            raise TypeError("provider must implement PaymentProvider")

        self.provider = provider
        self.repository = repository or InMemoryPaymentRepository()
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()
        self._lock = threading.RLock()

    def create_payment(
        self,
        request: PaymentRequest,
        *,
        idempotency_key: str,
        capture: bool = True,
    ) -> PaymentRecord:
        """Authorize a payment and optionally capture it immediately."""

        if not isinstance(request, PaymentRequest):
            raise TypeError("request must be a PaymentRequest")

        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint(
            {
                "operation": "authorize",
                "request": {
                    "amount": format(request.money.amount, "f"),
                    "currency": request.money.currency,
                    # Token is hashed rather than persisted in the fingerprint payload.
                    "token_sha256": hashlib.sha256(
                        request.payment_method_token.encode("utf-8")
                    ).hexdigest(),
                    "merchant_reference": request.merchant_reference,
                    "description": request.description,
                    "metadata": dict(request.metadata),
                },
                "capture": bool(capture),
            }
        )

        with self._lock:
            existing = self._resolve_idempotent(
                namespace="authorize",
                key=key,
                fingerprint=fingerprint,
            )
            if existing is not None:
                if capture and existing.remaining_capturable > 0:
                    return self.capture(
                        existing.payment_id,
                        idempotency_key=_derived_idempotency_key(key, "capture"),
                    )
                return existing

            try:
                authorization = self.provider.authorize(
                    request,
                    idempotency_key=key,
                )
            except PaymentError:
                raise
            except Exception as exc:
                raise PaymentProviderError("authorize", str(exc)) from exc

            now = datetime.now(timezone.utc)
            record = PaymentRecord(
                payment_id=uuid.uuid4().hex,
                provider_payment_id=authorization.provider_payment_id,
                merchant_reference=request.merchant_reference,
                currency=request.money.currency,
                authorized_amount=request.money.amount,
                status=PaymentStatus.AUTHORIZED,
                provider_reference=authorization.provider_reference,
                metadata={
                    **dict(request.metadata),
                    **dict(authorization.metadata),
                },
                created_at=now,
                updated_at=now,
            )
            self.repository.save(record)
            self.idempotency_store.put(
                "authorize",
                key,
                IdempotencyEntry(
                    fingerprint=fingerprint,
                    payment_id=record.payment_id,
                ),
            )

            logger.info(
                "Authorized payment id=%s reference=%s currency=%s",
                record.payment_id,
                record.merchant_reference,
                record.currency,
            )

        if capture:
            capture_key = _derived_idempotency_key(key, "capture")
            return self.capture(
                record.payment_id,
                idempotency_key=capture_key,
            )

        return self.get_payment(record.payment_id)

    def capture(
        self,
        payment_id: str,
        *,
        idempotency_key: str,
        amount: Optional[DecimalLike] = None,
    ) -> PaymentRecord:
        """Capture all or part of a previously authorized payment."""

        payment_id = _require_text(payment_id, "payment_id", max_length=200)
        key = _validate_idempotency_key(idempotency_key)

        with self._lock:
            record = self._require_record(payment_id)
            if record.status in {
                PaymentStatus.REFUNDED,
                PaymentStatus.VOIDED,
                PaymentStatus.FAILED,
            }:
                raise PaymentStateError(
                    f"Cannot capture payment in state '{record.status.value}'"
                )

            requested = (
                record.remaining_capturable
                if amount is None
                else _coerce_decimal(amount, "amount")
            )
            if requested <= 0:
                if record.remaining_capturable == 0:
                    return record
                raise PaymentValidationError("capture amount must be greater than zero")
            if requested > record.remaining_capturable:
                raise PaymentValidationError(
                    "capture amount exceeds remaining authorized amount"
                )

            fingerprint = _fingerprint(
                {
                    "operation": "capture",
                    "payment_id": payment_id,
                    "amount": format(requested, "f"),
                    "currency": record.currency,
                }
            )
            existing = self._resolve_idempotent(
                namespace="capture",
                key=key,
                fingerprint=fingerprint,
            )
            if existing is not None:
                return existing

            money = Money(requested, record.currency)
            try:
                result = self.provider.capture(
                    record.provider_payment_id,
                    money,
                    idempotency_key=key,
                )
            except PaymentError:
                raise
            except Exception as exc:
                raise PaymentProviderError("capture", str(exc)) from exc

            record.captured_amount += requested
            record.status = (
                PaymentStatus.CAPTURED
                if record.captured_amount == record.authorized_amount
                else PaymentStatus.PARTIALLY_CAPTURED
            )
            record.updated_at = datetime.now(timezone.utc)
            if result.provider_reference:
                record.provider_reference = result.provider_reference
            record.metadata.update(dict(result.metadata))
            record.metadata["last_capture_operation_id"] = result.operation_id

            self.repository.save(record)
            self.idempotency_store.put(
                "capture",
                key,
                IdempotencyEntry(fingerprint=fingerprint, payment_id=payment_id),
            )

            logger.info(
                "Captured payment id=%s amount=%s %s",
                payment_id,
                format(requested, "f"),
                record.currency,
            )
            return _clone_record(record)

    def refund(
        self,
        payment_id: str,
        *,
        idempotency_key: str,
        amount: Optional[DecimalLike] = None,
    ) -> PaymentRecord:
        """Refund all or part of the captured amount."""

        payment_id = _require_text(payment_id, "payment_id", max_length=200)
        key = _validate_idempotency_key(idempotency_key)

        with self._lock:
            record = self._require_record(payment_id)
            if record.captured_amount <= 0:
                raise PaymentStateError("Cannot refund a payment with no captured amount")
            if record.remaining_capturable > 0:
                raise PaymentStateError(
                    "Refund requires the authorization to be fully captured; "
                    "void/cancel the remaining authorization at the provider first"
                )

            requested = (
                record.remaining_refundable
                if amount is None
                else _coerce_decimal(amount, "amount")
            )
            if requested <= 0:
                if record.remaining_refundable == 0:
                    return record
                raise PaymentValidationError("refund amount must be greater than zero")
            if requested > record.remaining_refundable:
                raise PaymentValidationError(
                    "refund amount exceeds remaining refundable amount"
                )

            fingerprint = _fingerprint(
                {
                    "operation": "refund",
                    "payment_id": payment_id,
                    "amount": format(requested, "f"),
                    "currency": record.currency,
                }
            )
            existing = self._resolve_idempotent(
                namespace="refund",
                key=key,
                fingerprint=fingerprint,
            )
            if existing is not None:
                return existing

            money = Money(requested, record.currency)
            try:
                result = self.provider.refund(
                    record.provider_payment_id,
                    money,
                    idempotency_key=key,
                )
            except PaymentError:
                raise
            except Exception as exc:
                raise PaymentProviderError("refund", str(exc)) from exc

            record.refunded_amount += requested
            record.status = (
                PaymentStatus.REFUNDED
                if record.refunded_amount == record.captured_amount
                else PaymentStatus.PARTIALLY_REFUNDED
            )
            record.updated_at = datetime.now(timezone.utc)
            if result.provider_reference:
                record.provider_reference = result.provider_reference
            record.metadata.update(dict(result.metadata))
            record.metadata["last_refund_operation_id"] = result.operation_id

            self.repository.save(record)
            self.idempotency_store.put(
                "refund",
                key,
                IdempotencyEntry(fingerprint=fingerprint, payment_id=payment_id),
            )

            logger.info(
                "Refunded payment id=%s amount=%s %s",
                payment_id,
                format(requested, "f"),
                record.currency,
            )
            return _clone_record(record)

    def void(
        self,
        payment_id: str,
        *,
        idempotency_key: str,
    ) -> PaymentRecord:
        """Void a fully uncaptured authorization."""

        payment_id = _require_text(payment_id, "payment_id", max_length=200)
        key = _validate_idempotency_key(idempotency_key)

        with self._lock:
            record = self._require_record(payment_id)

            if record.status is PaymentStatus.VOIDED:
                return record
            if record.captured_amount > 0:
                raise PaymentStateError(
                    "Cannot void a payment after any amount has been captured"
                )
            if record.status is not PaymentStatus.AUTHORIZED:
                raise PaymentStateError(
                    f"Cannot void payment in state '{record.status.value}'"
                )

            fingerprint = _fingerprint(
                {
                    "operation": "void",
                    "payment_id": payment_id,
                    "currency": record.currency,
                    "authorized_amount": format(record.authorized_amount, "f"),
                }
            )
            existing = self._resolve_idempotent(
                namespace="void",
                key=key,
                fingerprint=fingerprint,
            )
            if existing is not None:
                return existing

            try:
                result = self.provider.void(
                    record.provider_payment_id,
                    idempotency_key=key,
                )
            except PaymentError:
                raise
            except Exception as exc:
                raise PaymentProviderError("void", str(exc)) from exc

            record.status = PaymentStatus.VOIDED
            record.updated_at = datetime.now(timezone.utc)
            if result.provider_reference:
                record.provider_reference = result.provider_reference
            record.metadata.update(dict(result.metadata))
            record.metadata["void_operation_id"] = result.operation_id

            self.repository.save(record)
            self.idempotency_store.put(
                "void",
                key,
                IdempotencyEntry(fingerprint=fingerprint, payment_id=payment_id),
            )
            logger.info("Voided payment id=%s", payment_id)
            return _clone_record(record)

    def get_payment(self, payment_id: str) -> PaymentRecord:
        payment_id = _require_text(payment_id, "payment_id", max_length=200)
        return self._require_record(payment_id)

    def _require_record(self, payment_id: str) -> PaymentRecord:
        record = self.repository.get(payment_id)
        if record is None:
            raise PaymentNotFoundError(payment_id)
        return record

    def _resolve_idempotent(
        self,
        *,
        namespace: str,
        key: str,
        fingerprint: str,
    ) -> Optional[PaymentRecord]:
        entry = self.idempotency_store.get(namespace, key)
        if entry is None:
            return None
        if entry.fingerprint != fingerprint:
            raise IdempotencyConflictError(key)
        return self._require_record(entry.payment_id)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _coerce_decimal(value: DecimalLike, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise PaymentValidationError(f"{field_name} must be numeric")
    if isinstance(value, float):
        raise PaymentValidationError(
            f"{field_name} must not be a float; use Decimal, int, or string"
        )
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PaymentValidationError(f"{field_name} is not a valid decimal") from exc


def _require_text(value: Any, field_name: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise PaymentValidationError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise PaymentValidationError(f"{field_name} must not be empty")
    if len(normalized) > max_length:
        raise PaymentValidationError(
            f"{field_name} exceeds maximum length of {max_length}"
        )
    return normalized


def _normalize_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise PaymentValidationError("metadata must be a mapping")
    normalized = {str(key): value for key, value in metadata.items()}
    try:
        json.dumps(normalized, sort_keys=True, default=str)
    except (TypeError, ValueError) as exc:
        raise PaymentValidationError("metadata must be serializable") from exc
    return normalized


def _validate_idempotency_key(key: str) -> str:
    if not isinstance(key, str):
        raise PaymentValidationError("idempotency_key must be a string")
    normalized = key.strip()
    if not _IDEMPOTENCY_RE.fullmatch(normalized):
        raise PaymentValidationError(
            "idempotency_key must be 8-200 characters and contain only "
            "letters, digits, '.', '_', ':', or '-'"
        )
    return normalized


def _clone_record(record: PaymentRecord) -> PaymentRecord:
    return replace(record, metadata=dict(record.metadata))


__all__ = [
    "AuthorizationResult",
    "IdempotencyConflictError",
    "IdempotencyEntry",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "InMemoryPaymentRepository",
    "Money",
    "PaymentError",
    "PaymentNotFoundError",
    "PaymentProvider",
    "PaymentProviderError",
    "PaymentRecord",
    "PaymentRepository",
    "PaymentRequest",
    "PaymentService",
    "PaymentStateError",
    "PaymentStatus",
    "PaymentValidationError",
    "ProviderOperationResult",
]


# -------------------------------------------------------------------------
# Temporary self-test block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    class _FakeProvider:
        def __init__(self) -> None:
            self.authorize_calls = 0
            self.capture_calls = 0
            self.refund_calls = 0

        def authorize(
            self,
            request: PaymentRequest,
            *,
            idempotency_key: str,
        ) -> AuthorizationResult:
            self.authorize_calls += 1
            return AuthorizationResult(
                provider_payment_id=f"provider_{self.authorize_calls}",
                provider_reference="auth_ref",
            )

        def capture(
            self,
            provider_payment_id: str,
            money: Money,
            *,
            idempotency_key: str,
        ) -> ProviderOperationResult:
            self.capture_calls += 1
            return ProviderOperationResult(
                operation_id=f"capture_{self.capture_calls}"
            )

        def refund(
            self,
            provider_payment_id: str,
            money: Money,
            *,
            idempotency_key: str,
        ) -> ProviderOperationResult:
            self.refund_calls += 1
            return ProviderOperationResult(
                operation_id=f"refund_{self.refund_calls}"
            )

        def void(
            self,
            provider_payment_id: str,
            *,
            idempotency_key: str,
        ) -> ProviderOperationResult:
            return ProviderOperationResult(operation_id="void_1")

    print("\n=== Running Payment self-test ===\n")
    provider = _FakeProvider()
    service = PaymentService(provider)

    request = PaymentRequest(
        money=Money.of("125.50", "EUR"),
        payment_method_token="tok_test_7f9d2642",
        merchant_reference="order-1001",
        metadata={"customer_id": "test-customer"},
    )

    payment = service.create_payment(
        request,
        idempotency_key="checkout:order-1001",
        capture=False,
    )
    assert payment.status is PaymentStatus.AUTHORIZED
    assert payment.authorized_amount == Decimal("125.50")

    duplicate = service.create_payment(
        request,
        idempotency_key="checkout:order-1001",
        capture=False,
    )
    assert duplicate.payment_id == payment.payment_id
    assert provider.authorize_calls == 1

    partial = service.capture(
        payment.payment_id,
        amount="25.50",
        idempotency_key="capture:order-1001:1",
    )
    assert partial.status is PaymentStatus.PARTIALLY_CAPTURED
    assert partial.captured_amount == Decimal("25.50")

    captured = service.capture(
        payment.payment_id,
        amount="100.00",
        idempotency_key="capture:order-1001:2",
    )
    assert captured.status is PaymentStatus.CAPTURED
    assert captured.captured_amount == Decimal("125.50")

    refunded = service.refund(
        payment.payment_id,
        amount="50.00",
        idempotency_key="refund:order-1001:1",
    )
    assert refunded.status is PaymentStatus.PARTIALLY_REFUNDED
    assert refunded.refunded_amount == Decimal("50.00")

    fully_refunded = service.refund(
        payment.payment_id,
        idempotency_key="refund:order-1001:2",
    )
    assert fully_refunded.status is PaymentStatus.REFUNDED
    assert fully_refunded.refunded_amount == Decimal("125.50")

    void_request = PaymentRequest(
        money=Money.of("15.00", "EUR"),
        payment_method_token="tok_void_8db31",
        merchant_reference="order-void",
    )
    voidable = service.create_payment(
        void_request,
        idempotency_key="checkout:order-void",
        capture=False,
    )
    voided = service.void(
        voidable.payment_id,
        idempotency_key="void:order-void:1",
    )
    assert voided.status is PaymentStatus.VOIDED

    try:
        PaymentRequest(
            money=Money.of("10", "EUR"),
            payment_method_token="4111 1111 1111 1111",
            merchant_reference="unsafe",
        )
    except PaymentValidationError:
        pass
    else:
        raise AssertionError("Raw card number must be rejected")

    print("✔ Payment self-test passed")
