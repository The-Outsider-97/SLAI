from __future__ import annotations
 
import threading
 
from typing import Dict, Optional, Type
from urllib.parse import urlparse
 
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_storage import AbstractStorage
from .local import LocalStorage
from .s3 import S3Storage
from .gcs import GCSStorage
from .azure import AzureStorage
from .http import HTTPStorage
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("Storage Factory")
printer = PrettyPrinter()
 
 
class StorageFactory:
    """Thread-safe registry and singleton cache for storage backends.
 
    Design
    ------
    * One instance per URI scheme is kept alive for the lifetime of the
      process (connection pooling, credential reuse).
    * ``register_backend`` allows downstream code to plug in custom backends
      without modifying this file.
    * ``clear_cache`` is provided for testing: it drops all cached instances
      so the next ``get_storage`` call creates fresh ones.
    * All mutations to ``_instances`` are protected by ``_lock``.
    """
 
    _backends: Dict[str, Type[AbstractStorage]] = {
        "file":  LocalStorage,
        "s3":    S3Storage,
        "gs":    GCSStorage,
        "azure": AzureStorage,
        "http":  HTTPStorage,
        "https": HTTPStorage,
    }
 
    _instances: Dict[str, AbstractStorage] = {}
    _lock: threading.Lock = threading.Lock()
 
    def __init__(self) -> None:
        self.config = load_global_config()
        self.factory_cfg: Dict = get_config_section("storage")
        self.default_backend: str = self.factory_cfg.get("default_backend", "s3")
        logger.debug({
            "event": "storage_factory_init",
            "registered_schemes": list(self._backends.keys()),
            "default_backend": self.default_backend,
        })
 
    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------
    @classmethod
    def register_backend(
        cls,
        scheme: str,
        backend_class: Type[AbstractStorage],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a custom storage backend for *scheme*.
 
        Parameters
        ----------
        scheme:
            URI scheme, e.g. ``"gcs"``, ``"minio"``.  Case-insensitive.
        backend_class:
            A concrete subclass of ``AbstractStorage``.
        overwrite:
            When ``False`` (default), raise ``DataConfigError`` if the scheme
            is already registered to prevent accidental silent replacements.
        """
        key = scheme.lower()
        with cls._lock:
            if key in cls._backends and not overwrite:
                raise DataConfigError(
                    f"Storage backend for scheme '{key}' is already registered. "
                    "Pass overwrite=True to replace it.",
                    context={"scheme": key, "existing": cls._backends[key].__name__},
                )
            cls._backends[key] = backend_class
            # Drop any cached instance so the new class is used on next access.
            cls._instances.pop(key, None)
        logger.info({"event": "backend_registered", "scheme": key, "class": backend_class.__name__})
 
    @classmethod
    def get_storage(cls, uri: str) -> AbstractStorage:
        """Return a (possibly cached) backend instance for *uri*.
 
        The scheme is extracted from *uri*.  Bare paths (no scheme) default to
        the ``"file"`` backend.
 
        Parameters
        ----------
        uri:
            Any scheme-qualified URI, e.g. ``'s3://bucket/key'``.
 
        Returns
        -------
        AbstractStorage
            A ready-to-use backend instance.
 
        Raises
        ------
        DataConfigError
            If no backend is registered for the detected scheme.
        """
        scheme = urlparse(uri).scheme or "file"
        key = scheme.lower()
 
        with cls._lock:
            if key not in cls._backends:
                raise DataConfigError(
                    f"No storage backend registered for scheme '{key}'",
                    context={
                        "scheme": key,
                        "uri": uri,
                        "available": sorted(cls._backends.keys()),
                    },
                )
            if key not in cls._instances:
                logger.info({"event": "backend_instantiated", "scheme": key})
                cls._instances[key] = cls._backends[key]()
            return cls._instances[key]
 
    @classmethod
    def clear_cache(cls) -> None:
        """Drop all cached backend instances.
 
        Intended for use in tests that need to swap in mock backends between
        test cases without module-level patching.
        """
        with cls._lock:
            cls._instances.clear()
        logger.debug({"event": "storage_cache_cleared"})
 
    @classmethod
    def registered_schemes(cls) -> list[str]:
        """Return a sorted list of currently registered URI schemes."""
        with cls._lock:
            return sorted(cls._backends.keys())
 
 
if __name__ == "__main__":
    print("\n=== Running factory ===\n")
    printer.status("TEST", "factory initialized", "info")
 
    StorageFactory.clear_cache()
 
    # Default schemes are registered
    schemes = StorageFactory.registered_schemes()
    for expected in ("file", "s3", "gs", "azure", "http", "https"):
        assert expected in schemes, f"Missing scheme: {expected}"
    printer.status("PASS", "All default schemes registered", "success")
 
    # get_storage returns correct types
    assert isinstance(StorageFactory.get_storage("file:///tmp/x"), LocalStorage)
    assert isinstance(StorageFactory.get_storage("/abs/path/to/file"), LocalStorage)
    assert isinstance(StorageFactory.get_storage("http://example.com/f"), HTTPStorage)
    assert isinstance(StorageFactory.get_storage("https://example.com/f"), HTTPStorage)
    printer.status("PASS", "get_storage returns correct backend types", "success")
 
    # Singleton — same instance returned on repeat calls
    inst_a = StorageFactory.get_storage("http://x.com/a")
    inst_b = StorageFactory.get_storage("http://x.com/b")
    assert inst_a is inst_b, "Expected singleton per scheme"
    printer.status("PASS", "Singleton per scheme enforced", "success")
 
    # Unknown scheme raises DataConfigError
    try:
        StorageFactory.get_storage("ftp://ftp.example.com/file")
        assert False
    except DataConfigError:
        printer.status("PASS", "Unknown scheme raises DataConfigError", "success")
 
    # Custom backend registration
    class _MockStorage(AbstractStorage):
        def open(self, uri, mode="rb", *, retry_config=None): ... # type: ignore
        def exists(self, uri): return False
        def list(self, prefix, recursive=False): return []
        def copy(self, src, dst, overwrite=False): ... # type: ignore
        def delete(self, uri): ...
 
    StorageFactory.register_backend("mock", _MockStorage)
    assert isinstance(StorageFactory.get_storage("mock://bucket/key"), _MockStorage)
    printer.status("PASS", "Custom backend registered and resolved", "success")
 
    # Duplicate registration without overwrite= raises
    try:
        StorageFactory.register_backend("mock", _MockStorage, overwrite=False)
        assert False
    except DataConfigError:
        printer.status("PASS", "Duplicate registration guarded", "success")
 
    # overwrite=True replaces silently
    StorageFactory.register_backend("mock", _MockStorage, overwrite=True)
    printer.status("PASS", "overwrite=True replaces existing backend", "success")
 
    # clear_cache drops cached instances
    StorageFactory.clear_cache()
    assert StorageFactory._instances == {}
    printer.status("PASS", "clear_cache empties instance cache", "success")
 
    print("\n=== Test ran successfully ===\n")