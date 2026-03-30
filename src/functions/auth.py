"""Authentication utilities for sign-up, login, and passcode-based verification flows."""

from __future__ import annotations

import hashlib
import secrets
import string

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Dict, Optional

from .utils.config_loader import get_config_section, load_global_config
from .utils.functions_error import (
    AccountLockedError,
    InvalidCredentialsError,
    InvalidTokenError,
    UserAlreadyExistsError,
)
from .functions_memory import CredentialPolicy, PasswordHasher, PortableStore, TTLCache
from logs.logger import PrettyPrinter, get_logger

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
class VerificationChallenge:
    challenge_id: str
    username: str
    code_hash: str
    purpose: str
    created_at: datetime
    expires_at: datetime
    used: bool = False
    used_at: Optional[datetime] = None


@dataclass
class UserRecord:
    user_id: str
    salt: str
    password_hash: str
    email: str = ""
    is_verified: bool = False
    failed_attempts: int = 0
    lockout_until: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    last_trusted_verification_at: Optional[datetime] = None


class AuthService:
    """Auth service with lockout, refresh rotation, revocation, and email challenge verification."""

    def __init__(
        self,
        token_ttl_minutes: int = 15,
        refresh_ttl_days: int = 14,
        max_failed_attempts: int = 5,
        lockout_minutes: int = 15,
        memory_path: str = "data/processed/functions_auth.json",
        pepper: Optional[str] = None,
        inactivity_hours: int = 24,
    ) -> None:
        self.config = load_global_config()
        self.auth_config = get_config_section("authentication")

        self._policy = CredentialPolicy()
        self._hasher = PasswordHasher(pepper=pepper)
        self._token_ttl = timedelta(minutes=token_ttl_minutes)
        self._refresh_ttl = timedelta(days=refresh_ttl_days)
        self._max_failed_attempts = max_failed_attempts
        self._lockout_minutes = lockout_minutes
        self._inactivity_delta = timedelta(hours=inactivity_hours)
        self._store = PortableStore(memory_path)

        self._users: Dict[str, UserRecord] = {}
        self._verification_challenges: Dict[str, VerificationChallenge] = {}
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

        logger.info("Authentication service successfully initialized")

    def _load_state(self) -> None:
        state = self._store.load()
        raw_users = state.get("users", {})
        for username, record in raw_users.items():
            lockout_until = record.get("lockout_until")
            self._users[username] = UserRecord(
                user_id=record["user_id"],
                salt=record["salt"],
                password_hash=record["password_hash"],
                email=record.get("email", ""),
                is_verified=record.get("is_verified", False),
                failed_attempts=record.get("failed_attempts", 0),
                lockout_until=datetime.fromisoformat(lockout_until) if lockout_until else None,
                last_activity_at=datetime.fromisoformat(record["last_activity_at"]) if record.get("last_activity_at") else None,
                last_trusted_verification_at=(
                    datetime.fromisoformat(record["last_trusted_verification_at"])
                    if record.get("last_trusted_verification_at")
                    else None
                ),
            )

        raw_challenges = state.get("verification_challenges", {})
        for cid, challenge in raw_challenges.items():
            self._verification_challenges[cid] = VerificationChallenge(
                challenge_id=challenge["challenge_id"],
                username=challenge["username"],
                code_hash=challenge["code_hash"],
                purpose=challenge["purpose"],
                created_at=datetime.fromisoformat(challenge["created_at"]),
                expires_at=datetime.fromisoformat(challenge["expires_at"]),
                used=challenge.get("used", False),
                used_at=datetime.fromisoformat(challenge["used_at"]) if challenge.get("used_at") else None,
            )

        raw_revoked = state.get("revoked", {})
        self._revoked = {h: datetime.fromisoformat(exp) for h, exp in raw_revoked.items()}
        self._purge_expired_revoked()
        self._purge_old_challenges()

    def _persist_state(self) -> None:
        with self._lock:
            payload = {
                "users": {
                    username: {
                        **asdict(record),
                        "lockout_until": record.lockout_until.isoformat() if record.lockout_until else None,
                        "last_activity_at": record.last_activity_at.isoformat() if record.last_activity_at else None,
                        "last_trusted_verification_at": (
                            record.last_trusted_verification_at.isoformat()
                            if record.last_trusted_verification_at
                            else None
                        ),
                    }
                    for username, record in self._users.items()
                },
                "verification_challenges": {
                    cid: {
                        **asdict(challenge),
                        "created_at": challenge.created_at.isoformat(),
                        "expires_at": challenge.expires_at.isoformat(),
                        "used_at": challenge.used_at.isoformat() if challenge.used_at else None,
                    }
                    for cid, challenge in self._verification_challenges.items()
                },
                "revoked": {h: exp.isoformat() for h, exp in self._revoked.items()},
            }
            self._store.save(payload)

    def _purge_expired_revoked(self) -> None:
        now = datetime.now(timezone.utc)
        for token_hash in [h for h, exp in self._revoked.items() if exp < now]:
            del self._revoked[token_hash]

    def _purge_old_challenges(self) -> None:
        now = datetime.now(timezone.utc)
        for cid in [
            challenge_id
            for challenge_id, challenge in self._verification_challenges.items()
            if challenge.expires_at < now or challenge.used
        ]:
            del self._verification_challenges[cid]

    @staticmethod
    def _hash_token_key(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_passcode(username: str, passcode: str) -> str:
        return hashlib.sha256(f"{username}::{passcode}".encode("utf-8")).hexdigest()

    def _is_locked_out(self, user: UserRecord) -> bool:
        return bool(user.lockout_until and datetime.now(timezone.utc) < user.lockout_until)

    def _generate_passcode(self, length: int = 24) -> str:
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{};:,.?/"
        while True:
            candidate = "".join(secrets.choice(alphabet) for _ in range(length))
            if (
                any(c.isalpha() for c in candidate)
                and any(c.isdigit() for c in candidate)
                and any(c in "!@#$%^&*()-_=+[]{};:,.?/" for c in candidate)
            ):
                return candidate

    def _get_user(self, username: str) -> UserRecord:
        user = self._users.get(username)
        if not user:
            raise InvalidCredentialsError("Invalid username or password")
        return user

    # --- Public Methods ---

    def sign_up(self, username: str, password: str, email: str = "") -> str:
        self._policy.validate(password)
        with self._lock:
            if username in self._users:
                raise UserAlreadyExistsError("Username already exists")
            salt, password_hash = self._hasher.hash_password(password)
            user = UserRecord(
                user_id=secrets.token_hex(12),
                salt=salt,
                password_hash=password_hash,
                email=email,
                is_verified=False,
            )
            self._users[username] = user
            self._persist_state()
        return user.user_id

    def verify_credentials(self, username: str, password: str) -> str:
        with self._lock:
            user = self._get_user(username)
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

            user.failed_attempts = 0
            user.lockout_until = None
            self._persist_state()
            return user.user_id

    def create_verification_challenge(
        self,
        username: str,
        purpose: str,
        expires_in_minutes: int = 30,
        invalidate_existing: bool = True,
    ) -> str:
        with self._lock:
            self._get_user(username)
            now = datetime.now(timezone.utc)
            if invalidate_existing:
                for challenge in self._verification_challenges.values():
                    if challenge.username == username and challenge.purpose == purpose and not challenge.used:
                        challenge.used = True
                        challenge.used_at = now

            passcode = self._generate_passcode(length=24)
            challenge_id = secrets.token_hex(10)
            self._verification_challenges[challenge_id] = VerificationChallenge(
                challenge_id=challenge_id,
                username=username,
                code_hash=self._hash_passcode(username, passcode),
                purpose=purpose,
                created_at=now,
                expires_at=now + timedelta(minutes=expires_in_minutes),
            )
            self._persist_state()
            return passcode

    def verify_passcode(self, username: str, passcode: str, purpose: Optional[str] = None) -> bool:
        with self._lock:
            self._get_user(username)
            now = datetime.now(timezone.utc)
            expected_hash = self._hash_passcode(username, passcode)

            candidates = [
                c
                for c in self._verification_challenges.values()
                if c.username == username and (purpose is None or c.purpose == purpose)
            ]
            candidates.sort(key=lambda item: item.created_at, reverse=True)

            for challenge in candidates:
                if challenge.used:
                    continue
                if challenge.expires_at < now:
                    challenge.used = True
                    challenge.used_at = now
                    continue
                if challenge.code_hash != expected_hash:
                    continue

                challenge.used = True
                challenge.used_at = now
                user = self._users[username]
                if challenge.purpose == "signup_verification":
                    user.is_verified = True
                user.last_trusted_verification_at = now
                user.last_activity_at = now
                self._persist_state()
                return True

            self._persist_state()
            return False

    def complete_login(self, username: str) -> AuthToken:
        with self._lock:
            user = self._get_user(username)
            now = datetime.now(timezone.utc)
            user.last_activity_at = now
            self._persist_state()
            return self._mint_session(user.user_id).access

    def requires_reverification(self, username: str) -> bool:
        with self._lock:
            user = self._get_user(username)
            if not user.last_activity_at:
                return False
            return datetime.now(timezone.utc) - user.last_activity_at >= self._inactivity_delta

    def record_activity(self, username: str) -> None:
        with self._lock:
            user = self._get_user(username)
            user.last_activity_at = datetime.now(timezone.utc)
            self._persist_state()

    def get_user_email(self, username: str) -> str:
        with self._lock:
            user = self._users.get(username)
            return user.email if user else ""

    def is_account_verified(self, username: str) -> bool:
        with self._lock:
            user = self._users.get(username)
            return bool(user and user.is_verified)

    # Backward compatibility methods
    def log_in(self, username: str, password: str) -> AuthToken:
        self.verify_credentials(username, password)
        return self.complete_login(username)

    def log_in_with_refresh(self, username: str, password: str) -> AuthSession:
        self.verify_credentials(username, password)
        token = self.complete_login(username)
        now = datetime.now(timezone.utc)
        refresh_raw = secrets.token_urlsafe(48)
        refresh_hash = self._hash_token_key(refresh_raw)
        self._refresh_tokens.set(
            refresh_hash,
            {"user_id": token.user_id, "expires_at": now + self._refresh_ttl},
        )
        return AuthSession(
            access=token,
            refresh=RefreshToken(token=refresh_raw, user_id=token.user_id, expires_at=now + self._refresh_ttl),
        )

    def revoke_token(self, token: str) -> None:
        token_hash = self._hash_token_key(token)
        with self._lock:
            expiry = None
            access = self._access_tokens.get(token_hash)
            if access:
                expiry = access["expires_at"]
            else:
                refresh = self._refresh_tokens.get(token_hash)
                if refresh:
                    expiry = refresh["expires_at"]
            if expiry is None:
                expiry = datetime.now(timezone.utc) + max(self._token_ttl, self._refresh_ttl)

            self._revoked[token_hash] = expiry
            self._access_tokens.invalidate(token_hash)
            self._refresh_tokens.invalidate(token_hash)
            self._persist_state()

    def rotate_refresh_token(self, refresh_token: str) -> AuthSession:
        old_hash = self._hash_token_key(refresh_token)
        with self._lock:
            if old_hash in self._revoked:
                raise InvalidTokenError("Refresh token revoked")

            stored = self._refresh_tokens.get(old_hash)
            if not stored:
                raise InvalidTokenError("Invalid or expired refresh token")

            self._revoked[old_hash] = stored["expires_at"]
            self._refresh_tokens.invalidate(old_hash)
            self._persist_state()
            return self._mint_session(stored["user_id"])

    def is_token_valid(self, token: str) -> bool:
        token_hash = self._hash_token_key(token)
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
        with self._lock:
            return dict(self._revoked)

    def _mint_session(self, user_id: str) -> AuthSession:
        now = datetime.now(timezone.utc)
        access_raw = secrets.token_urlsafe(32)
        refresh_raw = secrets.token_urlsafe(48)

        access_hash = self._hash_token_key(access_raw)
        refresh_hash = self._hash_token_key(refresh_raw)

        self._access_tokens.set(
            access_hash,
            {"user_id": user_id, "expires_at": now + self._token_ttl},
        )
        self._refresh_tokens.set(
            refresh_hash,
            {"user_id": user_id, "expires_at": now + self._refresh_ttl},
        )

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
