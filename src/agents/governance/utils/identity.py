"""Authenticated identity assertions for tenant- and purpose-bound access.

The ``IdentityAuthority`` is the only component that may mint a
``verified=True`` ``IdentityContext``. Assertions are HMAC-SHA256 signed,
carry an expiry and are bound to a tenant. Without a configured signing key the
authority reports itself unavailable and *every* verification fails closed.

Key management is deliberately external: the key is injected by the host or
read from ``SLAI_GOVERNANCE_IDENTITY_KEY``. This module never generates or
persists secrets.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from typing import Iterable, Optional

from ..processing_context import IdentityContext

__all__ = ["IdentityAuthority", "IdentityVerificationError", "ENV_KEY"]

ENV_KEY = "SLAI_GOVERNANCE_IDENTITY_KEY"
_MIN_KEY_BYTES = 32


class IdentityVerificationError(PermissionError):
    """``code`` is one of: missing, malformed, forged, expired, unavailable, tenant_mismatch."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class IdentityAuthority:
    def __init__(self, key: Optional[bytes] = None, *, env: Optional[dict] = None):
        if key is None:
            raw = (env if env is not None else os.environ).get(ENV_KEY)
            key = raw.encode("utf-8") if raw else None
        self._key = key if key and len(key) >= _MIN_KEY_BYTES else None

    @property
    def available(self) -> bool:
        return self._key is not None

    def issue(
        self,
        principal_id: str,
        tenant_id: Optional[str],
        *,
        actor_type: str = "service",
        roles: Iterable[str] = (),
        ttl_seconds: float = 900.0,
        now: Optional[float] = None,
    ) -> str:
        if not self.available:
            raise IdentityVerificationError("unavailable", "identity signing key not configured")
        if not principal_id:
            raise ValueError("principal_id is required")
        issued = time.time() if now is None else now
        payload = {
            "p": principal_id,
            "t": tenant_id,
            "a": actor_type,
            "r": sorted(set(roles)),
            "iat": issued,
            "exp": issued + float(ttl_seconds),
        }
        body = _b64e(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return f"{body}.{self._sign(body)}"

    def verify(
        self,
        token: Optional[str],
        *,
        expected_tenant: Optional[str] = None,
        now: Optional[float] = None,
    ) -> IdentityContext:
        if not token:
            raise IdentityVerificationError("missing")
        if not self.available:
            raise IdentityVerificationError("unavailable", "identity signing key not configured")
        try:
            body, signature = token.split(".", 1)
        except ValueError as exc:
            raise IdentityVerificationError("malformed") from exc
        if not hmac.compare_digest(signature, self._sign(body)):
            raise IdentityVerificationError("forged")
        try:
            payload = json.loads(_b64d(body))
            principal = str(payload["p"])
            expires = float(payload["exp"])
        except (ValueError, KeyError, TypeError) as exc:
            raise IdentityVerificationError("malformed") from exc
        current = time.time() if now is None else now
        if current >= expires:
            raise IdentityVerificationError("expired")
        tenant = payload.get("t")
        if expected_tenant is not None and tenant != expected_tenant:
            raise IdentityVerificationError("tenant_mismatch")
        return IdentityContext(
            principal_id=principal,
            tenant_id=tenant,
            verified=True,
            actor_type=str(payload.get("a", "service")),
            roles=tuple(payload.get("r", ())),
            expires_at=expires,
            token=token,
        )

    def _sign(self, body: str) -> str:
        assert self._key is not None
        return _b64e(hmac.new(self._key, body.encode("ascii"), hashlib.sha256).digest())
