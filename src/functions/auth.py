"""Authentication utilities for sign-up and log-in flows."""

from __future__ import annotations

import hashlib
import secrets

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Dict, Optional

from .utils.config_loader import load_global_config, get_config_section
from .utils.functions_error import (
    UserAlreadyExistsError,
    InvalidCredentialsError,
    AccountLockedError,
    InvalidTokenError,
)
from .functions_memory import CredentialPolicy, PasswordHasher, PortableStore, TTLCache
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Authentication Service")
printer = PrettyPrinter


@dataclass(frozen=True)
class AuthToken:
    token: str
    user_id: str
    expires_at: datetime


@dataclass(frozen=True)
class RefreshToken:
    token: str
    user_id: str
    expires_at: datetime


@dataclass(frozen=True)
class AuthSession:
    access: AuthToken
    refresh: RefreshToken


@dataclass
class UserRecord:
    user_id: str
    salt: str
    password_hash: str
    failed_attempts: int = 0
    lockout_until: Optional[datetime] = None


class AuthService:
    """Auth service with lockout, refresh rotation, and revocation support."""

    def __init__(
        self,
        token_ttl_minutes: int = 15,
        refresh_ttl_days: int = 14,
        max_failed_attempts: int = 5,
        lockout_minutes: int = 15,
        memory_path: str = "data/processed/functions_auth.json",
        pepper: Optional[str] = None,
    ) -> None:
        self.config = load_global_config()
        self.auth_config = get_config_section('authentication')

        self._policy = CredentialPolicy()
        self._hasher = PasswordHasher(pepper=pepper)
        self._token_ttl = timedelta(minutes=token_ttl_minutes)
        self._refresh_ttl = timedelta(days=refresh_ttl_days)
        self._max_failed_attempts = max_failed_attempts
        self._lockout_minutes = lockout_minutes
        self._store = PortableStore(memory_path)

        self._users: Dict[str, UserRecord] = {}
        self._access_tokens: TTLCache[dict] = TTLCache(
            max_size=20_000,
            ttl_seconds=int(self._token_ttl.total_seconds()),
        )
        self._refresh_tokens: TTLCache[dict] = TTLCache(
            max_size=20_000,
            ttl_seconds=int(self._refresh_ttl.total_seconds()),
        )
        self._revoked: Dict[str, datetime] = {}
        self._load_state()
        self._lock = RLock()

        logger.info(f"Authentication service successfully initialized")

    def _load_state(self) -> None:
        state = self._store.load()
        raw_users = state.get("users", {})
        for username, record in raw_users.items():
            lockout_until = record.get("lockout_until")
            self._users[username] = UserRecord(
                user_id=record["user_id"],
                salt=record["salt"],
                password_hash=record["password_hash"],
                failed_attempts=record.get("failed_attempts", 0),
                lockout_until=datetime.fromisoformat(lockout_until) if lockout_until else None,
            )
        raw_revoked = state.get("revoked", {})
        self._revoked = {
            h: datetime.fromisoformat(exp) for h, exp in raw_revoked.items()
        }
        self._purge_expired_revoked()

    def _persist_state(self) -> None:
        with self._lock:
            payload = {
                "users": {
                    username: {
                        **asdict(record),
                        "lockout_until": record.lockout_until.isoformat() if record.lockout_until else None,
                    }
                    for username, record in self._users.items()
                },
                "revoked": {h: exp.isoformat() for h, exp in self._revoked.items()},
            }
            self._store.save(payload)

    def _purge_expired_revoked(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [h for h, exp in self._revoked.items() if exp < now]
        for h in expired:
            del self._revoked[h]

    @staticmethod
    def _hash_token_key(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _is_locked_out(self, user: UserRecord) -> bool:
        return bool(user.lockout_until and datetime.now(timezone.utc) < user.lockout_until)

    # --- Public Methods ---

    def sign_up(self, username: str, password: str) -> str:
        self._policy.validate(password)  # no need lock for validation
        with self._lock:
            if username in self._users:
                raise UserAlreadyExistsError("Username already exists")
            salt, password_hash = self._hasher.hash_password(password)
            user = UserRecord(user_id=secrets.token_hex(12), salt=salt, password_hash=password_hash)
            self._users[username] = user
            self._persist_state()   # persist inside lock
        return user.user_id

    def log_in(self, username: str, password: str) -> AuthToken:
        return self.log_in_with_refresh(username, password).access

    def log_in_with_refresh(self, username: str, password: str) -> AuthSession:
        with self._lock:
            user = self._users.get(username)
            if not user:
                raise InvalidCredentialsError("Invalid username or password")

            # Reset lockout if it has expired
            if user.lockout_until and datetime.now(timezone.utc) >= user.lockout_until:
                user.failed_attempts = 0
                user.lockout_until = None
                self._persist_state()

            if self._is_locked_out(user):
                raise AccountLockedError("Account temporarily locked due to repeated failed attempts")

            if not self._hasher.verify_password(password, user.salt, user.password_hash):
                user.failed_attempts += 1
                if user.failed_attempts >= self._max_failed_attempts:
                    user.lockout_until = datetime.now(timezone.utc) + timedelta(minutes=self._lockout_minutes)
                self._persist_state()
                raise InvalidCredentialsError("Invalid username or password")

            # Success: reset counters and mint session
            user.failed_attempts = 0
            user.lockout_until = None
            self._persist_state()
            return self._mint_session(user.user_id)

    def revoke_token(self, token: str) -> None:
        token_hash = self._hash_token_key(token)
        with self._lock:
            # Try to get expiry from both caches
            expiry = None
            access = self._access_tokens.get(token_hash)
            if access:
                expiry = access["expires_at"]
            else:
                refresh = self._refresh_tokens.get(token_hash)
                if refresh:
                    expiry = refresh["expires_at"]
            if expiry is None:
                # Token not found in active caches – fallback to max TTL
                expiry = datetime.now(timezone.utc) + max(self._token_ttl, self._refresh_ttl)

            self._revoked[token_hash] = expiry
            self._access_tokens.invalidate(token_hash)
            self._refresh_tokens.invalidate(token_hash)
            self._persist_state()

    def rotate_refresh_token(self, refresh_token: str) -> AuthSession:
        old_hash = self._hash_token_key(refresh_token)
        with self._lock:
            # Check if already revoked
            if old_hash in self._revoked:
                raise InvalidTokenError("Refresh token revoked")

            stored = self._refresh_tokens.get(old_hash)
            if not stored:
                raise InvalidTokenError("Invalid or expired refresh token")

            # Revoke the old token
            self._revoked[old_hash] = stored["expires_at"]
            self._refresh_tokens.invalidate(old_hash)
            self._persist_state()
            return self._mint_session(stored["user_id"])

    def is_token_valid(self, token: str) -> bool:
        token_hash = self._hash_token_key(token)
        # Check revocation
        if token_hash in self._revoked:
            return False
        stored = self._access_tokens.get(token_hash)
        if not stored:
            return False
        return datetime.now(timezone.utc) <= stored["expires_at"]

    def is_refresh_token_valid(self, refresh_token: str) -> bool:
        token_hash = self._hash_token_key(refresh_token)
        if token_hash in self._revoked:
            return False
        stored = self._refresh_tokens.get(token_hash)
        if not stored:
            return False
        return datetime.now(timezone.utc) <= stored["expires_at"]

    def log_out(self, token: str) -> None:
        self.revoke_token(token)

    def get_revocation_list(self) -> Dict[str, datetime]:
        """Return a copy of the revoked tokens map (hash -> expiry)."""
        with self._lock:
            return dict(self._revoked)

    def _mint_session(self, user_id: str) -> AuthSession:
        now = datetime.now(timezone.utc)
        access_raw = secrets.token_urlsafe(32)
        refresh_raw = secrets.token_urlsafe(48)

        access_hash = self._hash_token_key(access_raw)
        refresh_hash = self._hash_token_key(refresh_raw)

        # Store only the hash + metadata in the cache
        self._access_tokens.set(
            access_hash,
            {"user_id": user_id, "expires_at": now + self._token_ttl}
        )
        self._refresh_tokens.set(
            refresh_hash,
            {"user_id": user_id, "expires_at": now + self._refresh_ttl}
        )

        # Return the raw tokens to the client
        access = AuthToken(token=access_raw, user_id=user_id, expires_at=now + self._token_ttl)
        refresh = RefreshToken(token=refresh_raw, user_id=user_id, expires_at=now + self._refresh_ttl)
        return AuthSession(access=access, refresh=refresh)


if __name__ == "__main__":
    print("\n=== Running Authentication service ===\n")
    printer.status("Init", "Authentication service initialized", "success")

    service = AuthService(
        token_ttl_minutes=15,
        refresh_ttl_days=14,
        max_failed_attempts=5,
        lockout_minutes=15,
        memory_path="data/processed/functions_auth.json",
        pepper=None,
    )
    print(service)

    print("\n* * * * * Phase 2 - Sign Up / Log In * * * * *\n")
    name = "JohnKelly99"
    password = "Q1az!Baby123"

    try:
        # Sign up first
        user_id = service.sign_up(username=name, password=password)
        print(f"User signed up: {user_id}")
    except UserAlreadyExistsError:
        print("User already exists, proceeding to login.")

    # Log in
    session = service.log_in_with_refresh(username=name, password=password)
    print(f"Access token: {session.access.token[:20]}...")
    print(f"Refresh token: {session.refresh.token[:20]}...")

    print("\n* * * * * Phase 3 - Log Out * * * * *\n")
    # Pass the raw token string, not the AuthToken object
    service.log_out(session.access.token)
    print("Logged out. Access token should be invalid now.")

    # Validate after logout
    valid = service.is_token_valid(session.access.token)
    print(f"Access token valid after logout? {valid}")

    print("\n=== Successfully ran the Authentication service ===\n")