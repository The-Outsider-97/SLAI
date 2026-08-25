"""Production-ready shared memory and security primitives for `src.functions`.

Includes:
- configurable credential policy validation
- scrypt password hashing with constant-time verification
- thread-safe TTL/LRU cache
- atomic JSON persistence with cross-platform file locking
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any, Dict, Generic, Iterable, Iterator, Mapping, Optional, Tuple, TypeVar

import portalocker  # type: ignore

from .utils.config_loader import get_config_section, load_global_config
from .utils.functions_error import (
    CacheConfigurationError,
    CredentialPolicyError,
    PasswordHashingError,
    StoreLoadError,
    StoreLockError,
    StoreSaveError,
    StoreSerializationError,
)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Functions Memory")
printer = PrettyPrinter

T = TypeVar("T")


@dataclass(frozen=True)
class CredentialPolicy:
    """Password policy with configurable validation rules."""

    min_length: int = 12
    require_upper: bool = True
    require_lower: bool = True
    require_digit: bool = True
    require_symbol: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.min_length, int) or self.min_length < 1:
            raise CredentialPolicyError("credential_policy.min_length must be an integer >= 1")

    @classmethod
    def from_config(cls) -> "CredentialPolicy":
        """Load credential policy from the shared YAML config."""
        config = get_config_section("credential_policy")
        return cls(
            min_length=int(config.get("min_length", 12)),
            require_upper=bool(config.get("require_upper", True)),
            require_lower=bool(config.get("require_lower", True)),
            require_digit=bool(config.get("require_digit", True)),
            require_symbol=bool(config.get("require_symbol", True)),
        )

    def validate(self, password: str) -> None:
        """Validate a password against the configured policy.

        Raises:
            CredentialPolicyError: if the password is invalid.
        """
        violations: list[str] = []
    
        if len(password) < self.min_length:
            violations.append(f"Password must be at least {self.min_length} characters")
        if self.require_upper and not any(c.isupper() for c in password):
            violations.append("Password must include an uppercase letter")
        if self.require_lower and not any(c.islower() for c in password):
            violations.append("Password must include a lowercase letter")
        if self.require_digit and not any(c.isdigit() for c in password):
            violations.append("Password must include a digit")
        if self.require_symbol and not any(not c.isalnum() for c in password):
            violations.append("Password must include a symbol")
    
        if violations:
            raise CredentialPolicyError(
                reason="Password failed credential policy validation",
                violations=violations,
            )

    def is_valid(self, password: str) -> bool:
        """Return True when the password satisfies the policy."""
        try:
            self.validate(password)
        except CredentialPolicyError:
            return False
        return True


class PasswordHasher:
    """Memory-hard scrypt hasher with constant-time verification."""

    def __init__(
        self,
        pepper: Optional[str] = None,
        n: Optional[int] = None,
        r: Optional[int] = None,
        p: Optional[int] = None,
        salt_bytes: Optional[int] = None,
        dklen: Optional[int] = None,
        maxmem: Optional[int] = None,
    ) -> None:
        config = get_config_section("password_hasher")

        self._pepper = pepper or ""
        self._n = int(n if n is not None else config.get("n", 2**14))
        self._r = int(r if r is not None else config.get("r", 8))
        self._p = int(p if p is not None else config.get("p", 1))
        self._salt_bytes = int(salt_bytes if salt_bytes is not None else config.get("salt_bytes", 16))
        self._dklen = int(dklen if dklen is not None else config.get("dklen", 64))
        self._maxmem = int(maxmem if maxmem is not None else config.get("maxmem", 0))

        self._validate_parameters()
        logger.debug(
            "PasswordHasher initialized with n=%s, r=%s, p=%s, salt_bytes=%s, dklen=%s",
            self._n,
            self._r,
            self._p,
            self._salt_bytes,
            self._dklen,
        )

    def _validate_parameters(self) -> None:
        if self._n <= 1 or (self._n & (self._n - 1)) != 0:
            raise PasswordHashingError("password_hasher.n must be a power of 2 greater than 1")
        if self._r < 1:
            raise PasswordHashingError("password_hasher.r must be >= 1")
        if self._p < 1:
            raise PasswordHashingError("password_hasher.p must be >= 1")
        if self._salt_bytes < 16:
            raise PasswordHashingError("password_hasher.salt_bytes must be >= 16")
        if self._dklen < 32:
            raise PasswordHashingError("password_hasher.dklen must be >= 32")
        if self._maxmem < 0:
            raise PasswordHashingError("password_hasher.maxmem must be >= 0")

    def _derive_hash(self, password: str, salt_hex: str) -> str:
        if not isinstance(password, str):
            raise PasswordHashingError("Password must be a string")
        if not isinstance(salt_hex, str) or not salt_hex:
            raise PasswordHashingError("Salt must be a non-empty hex string")

        try:
            salt = bytes.fromhex(salt_hex)
        except ValueError as exc:
            raise PasswordHashingError("Salt must be valid hexadecimal") from exc

        try:
            derived = hashlib.scrypt(
                (password + self._pepper).encode("utf-8"),
                salt=salt,
                n=self._n,
                r=self._r,
                p=self._p,
                maxmem=self._maxmem,
                dklen=self._dklen,
            )
        except (TypeError, ValueError) as exc:
            raise PasswordHashingError(f"Failed to derive password hash: {exc}") from exc

        return derived.hex()

    def hash_password(self, password: str) -> Tuple[str, str]:
        """Hash a password and return `(salt_hex, digest_hex)`."""
        salt_hex = secrets.token_hex(self._salt_bytes)
        digest_hex = self._derive_hash(password=password, salt_hex=salt_hex)
        return salt_hex, digest_hex

    def verify_password(self, password: str, salt: str, expected_hash: str) -> bool:
        """Verify a password using constant-time comparison.

        Invalid stored hash material is treated as a failed verification and logged.
        """
        if not isinstance(expected_hash, str) or not expected_hash:
            logger.warning("Password verification failed because the stored hash is empty or invalid")
            return False

        try:
            computed_hash = self._derive_hash(password=password, salt_hex=salt)
        except PasswordHashingError as exc:
            logger.warning("Password verification failed: %s", exc)
            return False

        return hmac.compare_digest(computed_hash, expected_hash)


class TTLCache(Generic[T]):
    """Thread-safe TTL + LRU cache with explicit invalidation and inspection."""

    def __init__(self, max_size: Optional[int] = None, ttl_seconds: Optional[int] = None) -> None:
        config = get_config_section("ttl_cache")
        self.max_size = int(max_size if max_size is not None else config.get("max_size", 512))
        self.ttl_seconds = int(ttl_seconds if ttl_seconds is not None else config.get("ttl_seconds", 300))

        if self.max_size < 1:
            raise CacheConfigurationError("ttl_cache.max_size must be >= 1")
        if self.ttl_seconds < 0:
            raise CacheConfigurationError("ttl_cache.ttl_seconds must be >= 0")

        self._data: "OrderedDict[str, Tuple[float, T]]" = OrderedDict()
        self._lock = RLock()
        logger.debug(
            "TTLCache initialized: max_size=%s, ttl_seconds=%s",
            self.max_size,
            self.ttl_seconds,
        )

    def _expiry_for_new_value(self) -> float:
        return monotonic() + float(self.ttl_seconds)

    def _purge_expired_unlocked(self) -> int:
        if not self._data:
            return 0

        now = monotonic()
        expired_keys = [key for key, (expires_at, _) in self._data.items() if expires_at <= now]
        for key in expired_keys:
            self._data.pop(key, None)
        return len(expired_keys)

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._purge_expired_unlocked()
            if key in self._data:
                self._data.pop(key, None)
            self._data[key] = (self._expiry_for_new_value(), value)
            self._data.move_to_end(key)

            while len(self._data) > self.max_size:
                evicted_key, _ = self._data.popitem(last=False)
                logger.debug("TTLCache evicted LRU key '%s'", evicted_key)

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            self._purge_expired_unlocked()
            entry = self._data.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if expires_at <= monotonic():
                self._data.pop(key, None)
                return None

            self._data.move_to_end(key)
            return value

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def pop(self, key: str, default: Optional[T] = None) -> Optional[T]:
        with self._lock:
            self._purge_expired_unlocked()
            entry = self._data.pop(key, None)
            return default if entry is None else entry[1]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def cleanup(self) -> int:
        with self._lock:
            return self._purge_expired_unlocked()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            expired = self._purge_expired_unlocked()
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "expired_removed": expired,
            }

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_unlocked()
            return len(self._data)


class PortableStore:
    """JSON-backed store with atomic writes and cross-platform file locking."""

    def __init__(
        self,
        path: str | Path,
        lock_timeout: Optional[float] = None,
        ensure_ascii: Optional[bool] = None,
        indent: Optional[int] = None,
        sort_keys: Optional[bool] = None,
        create_if_missing: Optional[bool] = None,
    ) -> None:
        config = get_config_section("portable_store")

        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.lock_timeout = float(lock_timeout if lock_timeout is not None else config.get("lock_timeout", 5.0))
        self.ensure_ascii = bool(ensure_ascii if ensure_ascii is not None else config.get("ensure_ascii", False))
        self.indent = int(indent if indent is not None else config.get("indent", 2))
        self.sort_keys = bool(sort_keys if sort_keys is not None else config.get("sort_keys", True))
        self.create_if_missing = bool(
            create_if_missing if create_if_missing is not None else config.get("create_if_missing", True)
        )

        if self.lock_timeout <= 0:
            raise StoreLockError("portable_store.lock_timeout must be > 0")
        if self.indent < 0:
            raise StoreSaveError("portable_store.indent must be >= 0")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = RLock()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            try:
                with portalocker.Lock(str(self.lock_path), mode="a+", timeout=self.lock_timeout):
                    yield
            except portalocker.exceptions.LockException as exc:
                logger.error("Failed to acquire lock for %s: %s", self.path, exc)
                raise StoreLockError(f"Failed to acquire lock for {self.path}: {exc}") from exc

    def _serialize_payload(self, payload: Mapping[str, object]) -> str:
        try:
            return json.dumps(
                payload,
                ensure_ascii=self.ensure_ascii,
                indent=self.indent,
                sort_keys=self.sort_keys,
            )
        except (TypeError, ValueError) as exc:
            raise StoreSerializationError(f"Payload for {self.path} is not JSON serializable: {exc}") from exc

    def save(self, payload: Mapping[str, object]) -> None:
        serialized = self._serialize_payload(payload)

        with self._locked():
            temp_path: Optional[Path] = None
            try:
                fd, temp_name = tempfile.mkstemp(
                    prefix=f"{self.path.stem}.",
                    suffix=".tmp",
                    dir=str(self.path.parent),
                )
                temp_path = Path(temp_name)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.path)
                logger.debug("Saved state to %s", self.path)
            except StoreSerializationError:
                raise
            except OSError as exc:
                logger.error("Failed to save state to %s: %s", self.path, exc)
                raise StoreSaveError(f"Failed to save store at {self.path}: {exc}") from exc
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    def load(self, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        fallback = {} if default is None else dict(default)

        if not self.path.exists():
            if self.create_if_missing:
                self.save(fallback)
            return fallback

        with self._locked():
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    raw = handle.read().strip()
                if not raw:
                    logger.warning("Store at %s is empty; returning default payload", self.path)
                    return fallback
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.error("Failed to decode JSON from %s: %s", self.path, exc)
                raise StoreLoadError(f"Store at {self.path} contains invalid JSON: {exc}") from exc
            except OSError as exc:
                logger.error("Failed to read store %s: %s", self.path, exc)
                raise StoreLoadError(f"Failed to read store at {self.path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise StoreLoadError(
                f"Store at {self.path} must deserialize to a JSON object, got {type(payload).__name__}"
            )

        logger.debug("Loaded state from %s", self.path)
        return payload

    def delete(self, missing_ok: bool = True) -> None:
        with self._locked():
            try:
                self.path.unlink(missing_ok=missing_ok)
                logger.debug("Deleted store file %s", self.path)
            except OSError as exc:
                logger.error("Failed to delete store %s: %s", self.path, exc)
                raise StoreSaveError(f"Failed to delete store at {self.path}: {exc}") from exc

    def export_items(self, items: Iterable[Tuple[str, object]]) -> None:
        self.save({key: value for key, value in items})


if __name__ == "__main__":
    print("\n=== Running Functions Memory ===\n")
    printer.status("TEST", "Functions Memory initialized", "info")

    try:
        config = load_global_config()
        printer.status(
            "CONFIG",
            f"Loaded config from {config.get('__config_path__', 'unknown')}",
            "success",
        )

        policy = CredentialPolicy.from_config()
        printer.status(
            "POLICY",
            f"Credential policy loaded (min_length={policy.min_length})",
            "success",
        )

        password = "StrongP@ssw0rd123!"
        policy.validate(password)
        printer.status("PASSWORD", "Credential policy validation passed", "success")

        hasher = PasswordHasher(pepper=os.getenv("FUNCTIONS_MEMORY_TEST_PEPPER", ""))
        salt, expected_hash = hasher.hash_password(password=password)
        assert hasher.verify_password(password=password, salt=salt, expected_hash=expected_hash) is True
        assert hasher.verify_password(password="wrong-password", salt=salt, expected_hash=expected_hash) is False
        printer.status("HASH", "Password hashing and verification passed", "success")

        cache = TTLCache[str](max_size=2, ttl_seconds=60)
        cache.set("alpha", "A")
        cache.set("beta", "B")
        assert cache.get("alpha") == "A"
        cache.set("gamma", "C")
        assert cache.get("beta") is None
        assert cache.get("alpha") == "A"
        assert cache.get("gamma") == "C"
        cache.clear()
        assert len(cache) == 0
        printer.status("CACHE", "TTL cache operations passed", "success")

        test_dir = Path.cwd() / "data" / "processed" / "functions_memory_test"
        store = PortableStore(test_dir / "state.json")
        payload = {
            "message": "hello world",
            "count": 3,
            "items": ["alpha", "beta", "gamma"],
        }
        store.save(payload)
        loaded_payload = store.load()
        assert loaded_payload == payload

        store.export_items([("alpha", 1), ("beta", {"nested": True})])
        exported_payload = store.load()
        assert exported_payload == {"alpha": 1, "beta": {"nested": True}}
        store.delete()
        printer.status("STORE", "Portable store save/load/export passed", "success")

    except Exception as exc:
        printer.status("FAIL", f"Functions Memory test failed: {exc}", "error")
        logger.exception("Functions Memory standalone test failed")
        raise

    print("\n=== Test ran successfully ===\n")