# data/storage/base.py
from __future__ import annotations

import abc
 
from typing import BinaryIO, Optional, List, Dict, Any

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Base Data Storage")
printer = PrettyPrinter()


class StorageFile(abc.ABC):
    """Abstract file handle — supports streaming reads and context management.
 
    Subclasses must implement ``read``, ``seek``, ``tell``, and ``close``.
    The ``readable`` / ``writable`` flags follow the ``io.RawIOBase`` convention
    so that consumers can pass a ``StorageFile`` wherever a ``BinaryIO`` is
    accepted.
    """
 
    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.file_cfg = get_config_section("storage_file")
        logger.debug({"event": "storage_file_init", "class": type(self).__name__})
 
    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
 
    @abc.abstractmethod
    def read(self, size: int = -1) -> bytes:
        """Read up to *size* bytes; -1 reads until EOF."""
 
    @abc.abstractmethod
    def seek(self, offset: int, whence: int = 0) -> int:
        """Change the stream position; return the new absolute position."""
 
    @abc.abstractmethod
    def tell(self) -> int:
        """Return the current stream position."""
 
    @abc.abstractmethod
    def close(self) -> None:
        """Release the underlying resource."""
 
    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------
 
    def __enter__(self) -> "StorageFile":
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
 
    # ------------------------------------------------------------------
    # Capability flags (io.RawIOBase convention)
    # ------------------------------------------------------------------
 
    def readable(self) -> bool:
        return True
 
    def writable(self) -> bool:
        return False
 
    def seekable(self) -> bool:
        """Override in subclasses that support arbitrary seeks."""
        return False
 
 
class AbstractStorage(abc.ABC):
    """Backend-agnostic storage interface.
 
    Every concrete backend (S3, GCS, Azure, Local, HTTP) must implement the
    five abstract methods below.  The interface is intentionally minimal:
    higher-level concerns (caching, validation, hashing) live in the pipeline
    layer, not here.
    """
 
    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.abstract_cfg = get_config_section("storage")
        logger.debug({"event": "abstract_storage_init", "class": type(self).__name__})
 
    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
 
    @abc.abstractmethod
    def open(self, uri: str, mode: str = "rb", *,
             retry_config: Optional[Dict[str, Any]] = None) -> StorageFile:
        """Open a remote/local object for binary reading.
 
        Parameters
        ----------
        uri:
            Scheme-qualified path, e.g. ``'s3://bucket/key'``,
            ``'file:///abs/path'``, ``'azure://container/blob'``.
        mode:
            Only ``'rb'`` is supported in the current release.
        retry_config:
            Optional per-call override for the retry policy defined in
            ``data_config.yaml → storage.retry``.
 
        Returns
        -------
        StorageFile
            An open handle usable as a context manager.
 
        Raises
        ------
        DataSourceError
            If the object cannot be opened (not found, permission denied, …).
        DataConfigError
            If the URI scheme is invalid or the backend is misconfigured.
        """
 
    @abc.abstractmethod
    def exists(self, uri: str) -> bool:
        """Return ``True`` if the object exists and is readable."""
 
    @abc.abstractmethod
    def list(self, prefix: str, recursive: bool = False) -> List[str]:
        """List all URIs that share *prefix*.
 
        Parameters
        ----------
        prefix:
            A scheme-qualified prefix, e.g. ``'gs://bucket/folder/'``.
        recursive:
            When ``True``, descend into sub-directories / virtual prefixes.
 
        Returns
        -------
        list[str]
            Sorted list of fully-qualified URIs.
        """
 
    @abc.abstractmethod
    def copy(self, src_uri: str, dst_uri: str, overwrite: bool = False) -> None:
        """Server-side copy from *src_uri* to *dst_uri*.
 
        Parameters
        ----------
        overwrite:
            When ``False`` (default), raise ``DataSourceError`` if the
            destination already exists.
        """
 
    @abc.abstractmethod
    def delete(self, uri: str) -> None:
        """Delete the object at *uri*.  Silent no-op if already absent."""
 

if __name__ == "__main__":
    print("\n=== Running base_storage ===\n")
    printer.status("TEST", "base_storage initialized", "info")
 
    # Verify abstract enforcement — instantiation must be rejected.
    try:
        AbstractStorage()  # type: ignore[abstract]
        assert False, "AbstractStorage must not be instantiable"
    except TypeError:
        printer.status("PASS", "AbstractStorage correctly prevents direct instantiation", "success")
 
    try:
        StorageFile()  # type: ignore[abstract]
        assert False, "StorageFile must not be instantiable"
    except TypeError:
        printer.status("PASS", "StorageFile correctly prevents direct instantiation", "success")
 
    # Verify capability flags on a minimal concrete subclass.
    class _DummyFile(StorageFile):
        def read(self, size=-1): return b""
        def seek(self, offset, whence=0): return 0
        def tell(self): return 0
        def close(self): pass
 
    f = _DummyFile()
    assert f.readable() is True
    assert f.writable() is False
    assert f.seekable() is False
    printer.status("PASS", "StorageFile capability flags correct", "success")
 
    # Verify context-manager protocol.
    with _DummyFile() as fh:
        assert fh.readable()
    printer.status("PASS", "StorageFile context manager protocol works", "success")
 
    print("\n=== Test ran successfully ===\n")