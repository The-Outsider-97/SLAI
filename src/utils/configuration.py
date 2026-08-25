"""Shared, side-effect-free YAML configuration infrastructure.

Importing this module performs no file access, logging setup, directory
creation, or background work. Configuration is read only when a caller invokes
``ConfigBinding.load`` (normally through a subsystem compatibility facade).
"""

from __future__ import annotations

import copy
import time

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import yaml # pyright: ignore[reportMissingModuleSource]


DEFAULT_CACHE_TTL_SECONDS = 60.0


@dataclass(frozen=True)
class ConfigCacheInfo:
    config_path: str | None
    mtime_ns: int | None
    size: int | None
    loaded_at: float | None
    age_seconds: float | None
    has_data: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "mtime": None if self.mtime_ns is None else self.mtime_ns / 1_000_000_000,
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "loaded_at": self.loaded_at,
            "age_seconds": self.age_seconds,
            "has_data": self.has_data,
        }


@dataclass
class _CacheEntry:
    data: dict[str, Any]
    mtime_ns: int
    size: int
    loaded_at: float
    loaded_monotonic: float


class DuplicateConfigKeyError(ValueError):
    """Raised when one YAML mapping declares the same key more than once."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)

        if key in mapping:
            line = key_node.start_mark.line + 1
            raise DuplicateConfigKeyError(
                f"Duplicate configuration key {key!r} at line {line}"
            )

        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class ConfigRepository:
    """Thread-safe, path-keyed YAML cache shared by all configuration facades."""

    def __init__(self) -> None:
        self._entries: dict[Path, _CacheEntry] = {}
        self._lock = RLock()

    @staticmethod
    def resolve(path: str | Path) -> Path:
        return Path(path).expanduser().resolve()

    def load(
        self,
        path: str | Path,
        *,
        force_reload: bool = False,
        cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> dict[str, Any]:
        resolved = self.resolve(path)
        with self._lock:
            stat_result = resolved.stat()
            entry = self._entries.get(resolved)
            expired = bool(
                entry is not None
                and cache_ttl > 0
                and (time.monotonic() - entry.loaded_monotonic) > cache_ttl
            )
            stale = bool(
                entry is None
                or force_reload
                or expired
                or entry.mtime_ns != stat_result.st_mtime_ns
                or entry.size != stat_result.st_size
            )

            if stale:
                with resolved.open("r", encoding="utf-8") as stream:
                    loaded = yaml.load(stream, Loader=UniqueKeySafeLoader) or {}
                if not isinstance(loaded, dict):
                    raise TypeError(
                        f"Config file {resolved} must contain a YAML mapping, "
                        f"got {type(loaded).__name__}"
                    )
                data = dict(loaded)
                data["__config_path__"] = str(resolved)
                entry = _CacheEntry(
                    data=data,
                    mtime_ns=stat_result.st_mtime_ns,
                    size=stat_result.st_size,
                    loaded_at=time.time(),
                    loaded_monotonic=time.monotonic(),
                )
                self._entries[resolved] = entry

            # Configuration is shared as a source, not as mutable runtime state.
            assert entry is not None
            return copy.deepcopy(entry.data)

    def section(
        self,
        path: str | Path,
        section_name: str,
        *,
        config: Mapping[str, Any] | None = None,
        default: Mapping[str, Any] | None = None,
        force_reload: bool = False,
        cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> dict[str, Any]:
        source = (
            config
            if config is not None
            else self.load(path, force_reload=force_reload, cache_ttl=cache_ttl)
        )
        section = source.get(section_name)
        if section is None:
            return copy.deepcopy(dict(default or {}))
        if not isinstance(section, Mapping):
            return copy.deepcopy(dict(default or {}))
        return copy.deepcopy(dict(section))

    def clear(self, path: str | Path | None = None) -> None:
        with self._lock:
            if path is None:
                self._entries.clear()
            else:
                self._entries.pop(self.resolve(path), None)

    def cache_info(self, path: str | Path) -> ConfigCacheInfo:
        resolved = self.resolve(path)
        with self._lock:
            entry = self._entries.get(resolved)
            if entry is None:
                return ConfigCacheInfo(None, None, None, None, None, False)
            age = time.monotonic() - entry.loaded_monotonic
            return ConfigCacheInfo(
                config_path=str(resolved),
                mtime_ns=entry.mtime_ns,
                size=entry.size,
                loaded_at=entry.loaded_at,
                age_seconds=age,
                has_data=True,
            )


DEFAULT_CONFIG_REPOSITORY = ConfigRepository()


class ConfigBinding:
    """Bind the shared repository to one subsystem's default YAML path."""

    def __init__(self, default_path: str | Path, repository: ConfigRepository | None = None) -> None:
        self.default_path = Path(default_path)
        self.repository = repository or DEFAULT_CONFIG_REPOSITORY
        self._lock = RLock()
        self._known_paths: set[Path] = set()
        self._last_path = self.default_path

    def _path(self, config_path: str | Path | None) -> Path:
        return self.default_path if config_path is None else Path(config_path)

    def load(
        self,
        config_path: str | Path | None = None,
        *,
        force_reload: bool = False,
        cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> dict[str, Any]:
        resolved = self.repository.resolve(self._path(config_path))
        with self._lock:
            self._known_paths.add(resolved)
            self._last_path = resolved
        return self.repository.load(
            resolved,
            force_reload=force_reload,
            cache_ttl=cache_ttl,
        )

    def reload(self, config_path: str | Path | None = None) -> dict[str, Any]:
        return self.load(config_path, force_reload=True)

    def section(
        self,
        section_name: str,
        config: Mapping[str, Any] | None = None,
        *,
        config_path: str | Path | None = None,
        default: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = config if config is not None else self.load(config_path)
        return self.repository.section(
            self._path(config_path),
            section_name,
            config=source,
            default=default,
        )

    def clear(self) -> None:
        with self._lock:
            paths = tuple(self._known_paths or {self.default_path})
            self._last_path = self.default_path
        for path in paths:
            self.repository.clear(path)

    def cache_info(self) -> dict[str, Any]:
        with self._lock:
            path = self._last_path
        return self.repository.cache_info(path).to_dict()


def bind_config(default_path: str | Path) -> ConfigBinding:
    """Create a lightweight compatibility binding without reading the file."""

    return ConfigBinding(default_path)
