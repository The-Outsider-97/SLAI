"""Shared memory/security primitives for `src.functions`.

Includes:
- strong password hashing utilities
- credential policy checks
- lightweight TTL/LRU cache
- portable JSON persistence helpers with file locking
"""

from __future__ import annotations

import secrets
import hashlib
import hmac
import json
import os
import portalocker

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Generic, Iterable, Optional, Tuple, TypeVar

from .utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Functions Memory")
printer = PrettyPrinter

T = TypeVar("T")


@dataclass(frozen=True)
class CredentialPolicy:
    """Password policy with configurable rules."""
    min_length: int = 12
    require_upper: bool = True
    require_lower: bool = True
    require_digit: bool = True
    require_symbol: bool = True

    @classmethod
    def from_config(cls) -> "CredentialPolicy":
        """Load policy from config (section 'credential_policy')."""
        config = get_config_section('credential_policy')
        return cls(
            min_length=config.get('min_length', 12),
            require_upper=config.get('require_upper', True),
            require_lower=config.get('require_lower', True),
            require_digit=config.get('require_digit', True),
            require_symbol=config.get('require_symbol', True),
        )

    def validate(self, password: str) -> None:
        if len(password) < self.min_length:
            raise ValueError(f"Password must be at least {self.min_length} characters")
        if self.require_upper and not any(c.isupper() for c in password):
            raise ValueError("Password must include an uppercase letter")
        if self.require_lower and not any(c.islower() for c in password):
            raise ValueError("Password must include a lowercase letter")
        if self.require_digit and not any(c.isdigit() for c in password):
            raise ValueError("Password must include a digit")
        if self.require_symbol and not any(not c.isalnum() for c in password):
            raise ValueError("Password must include a symbol")


class PasswordHasher:
    """Memory-hard scrypt hasher with constant-time verification.
    
    Parameters can be set via config or constructor.
    """
    def __init__(self, pepper: Optional[str] = None,
                 n: Optional[int] = None,
                 r: Optional[int] = None,
                 p: Optional[int] = None) -> None:
        # Load config once
        config = get_config_section('password_hasher')
        self._pepper = pepper or ""   # pepper should come from environment, not config!
        self._n = n if n is not None else config.get('n', 2**14)
        self._r = r if r is not None else config.get('r', 8)
        self._p = p if p is not None else config.get('p', 1)
        logger.debug(f"PasswordHasher initialized with n={self._n}, r={self._r}, p={self._p}")

    def hash_password(self, password: str) -> Tuple[str, str]:
        salt = secrets.token_hex(16)
        digest = hashlib.scrypt(
            (password + self._pepper).encode("utf-8"),
            salt=salt.encode("utf-8"),
            n=self._n,
            r=self._r,
            p=self._p,
        )
        return salt, digest.hex()

    def verify_password(self, password: str, salt: str, expected_hash: str) -> bool:
        computed = hashlib.scrypt(
            (password + self._pepper).encode("utf-8"),
            salt=salt.encode("utf-8"),
            n=self._n,
            r=self._r,
            p=self._p,
        ).hex()
        return hmac.compare_digest(computed, expected_hash)


class TTLCache(Generic[T]):
    """Thread-safe TTL + LRU cache with configurable limits."""
    def __init__(self, max_size: Optional[int] = None, ttl_seconds: Optional[int] = None) -> None:
        config = get_config_section('ttl_cache')
        self.max_size = max_size if max_size is not None else config.get('max_size', 512)
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else config.get('ttl_seconds', 300)
        self.ttl = timedelta(seconds=self.ttl_seconds)
        self._data: "OrderedDict[str, Tuple[datetime, T]]" = OrderedDict()
        self._lock = RLock()
        logger.debug(f"TTLCache initialized: max_size={self.max_size}, ttl={self.ttl_seconds}s")

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._purge_expired()
            self._data[key] = (datetime.now(timezone.utc), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            self._purge_expired()
            if key not in self._data:
                return None
            created_at, value = self._data[key]
            if datetime.now(timezone.utc) - created_at > self.ttl:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def _purge_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [k for k, (created_at, _) in self._data.items() if now - created_at > self.ttl]
        for key in expired:
            self._data.pop(key, None)


class PortableStore:
    """JSON store with atomic writes and cross‑platform file locking."""
    def __init__(self, path: str, lock_timeout: float = 5.0) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.lock_timeout = lock_timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = RLock()   # For thread safety within the same process

    def _acquire_lock(self) -> None:
        """Acquire an exclusive lock on the lock file."""
        # Use portalocker with a timeout
        self.lock_fd = open(self.lock_path, 'w')
        portalocker.lock(self.lock_fd, portalocker.LOCK_EX, timeout=self.lock_timeout)

    def _release_lock(self) -> None:
        """Release the lock and close the lock file."""
        if hasattr(self, 'lock_fd'):
            portalocker.unlock(self.lock_fd)
            self.lock_fd.close()
            try:
                self.lock_path.unlink()
            except OSError:
                pass

    def save(self, payload: Dict[str, object]) -> None:
        """Write payload atomically with a temporary file."""
        with self._thread_lock:   # Ensure thread safety within the process
            self._acquire_lock()
            try:
                temp = self.path.with_suffix(self.path.suffix + ".tmp")
                with temp.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                os.replace(temp, self.path)
                logger.debug(f"Saved state to {self.path}")
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                raise
            finally:
                self._release_lock()

    def load(self) -> Dict[str, object]:
        """Load payload with a shared lock."""
        if not self.path.exists():
            return {}
        with self._thread_lock:
            self._acquire_lock()
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                logger.debug(f"Loaded state from {self.path}")
                return data
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return {}
            finally:
                self._release_lock()

    def export_items(self, items: Iterable[Tuple[str, object]]) -> None:
        self.save({key: value for key, value in items})


if __name__ == "__main__":
    print("\n=== Running Functions Memory ===\n")
    printer.status("Init", "Functions Memory initialized", "success")

    password = "1Q23w45!"

    hasher = PasswordHasher(pepper=None)
    salt, expected_hash = hasher.hash_password(password=password)
    verify = hasher.verify_password(
        password=password,
        salt=salt,
        expected_hash=expected_hash,
    )
    print(hasher)
    print(verify)

