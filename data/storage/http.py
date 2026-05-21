from __future__ import annotations
 
import requests  # type: ignore
 
from typing import Dict, Any, List, Optional
from requests.adapters import HTTPAdapter  # type: ignore
from urllib3.util.retry import Retry  # type: ignore
 
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_storage import AbstractStorage, StorageFile
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("HTTP Storage")
printer = PrettyPrinter()
 
_VALID_SCHEMES = ("http://", "https://")
 
 
class HTTPFile(StorageFile):
    """Streaming HTTP response reader backed by ``requests.Response``.
 
    HTTP does not support seeking without Range request support on the server.
    ``tell()`` tracks bytes consumed so far; ``seek(0, 0)`` raises
    ``NotImplementedError`` to signal the caller to re-open the resource.
 
    Data is pulled from the ``iter_content`` iterator in chunks of
    ``chunk_size`` bytes (configured via ``data_config.yaml → storage.http``).
    Partial reads are buffered internally so that ``read(n)`` can satisfy
    requests smaller than one network chunk.
    """
 
    def __init__(self, response: requests.Response, uri: str, chunk_size: int = 8192) -> None:
        super().__init__()
        self.file_cfg = get_config_section("storage_file")
        self._response = response
        self._uri = uri
        self._chunk_size = chunk_size
        self._iter = response.iter_content(chunk_size=chunk_size)
        self._buffer = b""
        self._pos: int = 0
        self._closed: bool = False
 
    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
 
    def _pull_chunk(self) -> bytes:
        """Pull the next chunk from the HTTP stream; returns ``b""`` at EOF."""
        try:
            return next(self._iter)
        except StopIteration:
            return b""
        except requests.RequestException as exc:
            raise DataSourceError(
                f"HTTP stream read error: {self._uri}",
                context={"uri": self._uri},
                cause=exc,
            ) from exc
 
    # ------------------------------------------------------------------
    # StorageFile interface
    # ------------------------------------------------------------------
 
    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise DataSourceError("Read on closed HTTP file", context={"uri": self._uri})
 
        if size == 0:
            return b""
 
        if size < 0:
            # Drain the stream completely.
            chunks = [self._buffer]
            while True:
                chunk = self._pull_chunk()
                if not chunk:
                    break
                chunks.append(chunk)
            data = b"".join(chunks)
            self._buffer = b""
            self._pos += len(data)
            return data
 
        # Read exactly *size* bytes (or fewer at EOF).
        out = bytearray()
        while len(out) < size:
            needed = size - len(out)
            if self._buffer:
                take = min(needed, len(self._buffer))
                out.extend(self._buffer[:take])
                self._buffer = self._buffer[take:]
            else:
                chunk = self._pull_chunk()
                if not chunk:
                    break
                self._buffer = chunk
 
        self._pos += len(out)
        return bytes(out)
 
    def seek(self, offset: int, whence: int = 0) -> int:
        raise NotImplementedError(
            "HTTPFile does not support seek; close and re-open the URI to restart."
        )
 
    def tell(self) -> int:
        if self._closed:
            raise DataSourceError("Tell on closed HTTP file", context={"uri": self._uri})
        return self._pos
 
    def close(self) -> None:
        if not self._closed:
            self._response.close()
            self._closed = True
 
 
class HTTPStorage(AbstractStorage):
    """HTTP/HTTPS read-only backend.
 
    ``list``, ``copy``, and ``delete`` are intentionally unsupported: HTTP has
    no standard directory-listing or mutation protocol.
 
    A ``requests.Session`` with a ``urllib3`` retry strategy handles transient
    5xx errors and rate-limiting (429) automatically at the transport layer, so
    the ``@with_retry`` decorator is not applied here — it would double-count
    retries.
 
    Configuration is read from ``data_config.yaml → storage.http``.
    """
 
    def __init__(self, user_agent: Optional[str] = None) -> None:
        super().__init__()
        self.storage_cfg = get_config_section("storage")
        self.http_cfg: Dict[str, Any] = self.storage_cfg.get("http", {})
 
        self.timeout: float = float(self.http_cfg.get("timeout", 30.0))
        self.max_retries: int = int(self.http_cfg.get("max_retries", 3))
        self.chunk_size: int = int(self.http_cfg.get("chunk_size", 8192))
        _user_agent: str = user_agent or self.http_cfg.get("user_agent", "DataLoader/1.0")
 
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": _user_agent})
 
        logger.info({
            "event": "http_storage_init",
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "user_agent": _user_agent,
        })
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _validate_uri(uri: str) -> None:
        if not uri.startswith(_VALID_SCHEMES):
            raise DataSourceError(
                "HTTPStorage only supports http:// and https:// URIs",
                context={"uri": uri},
            )
 
    # ------------------------------------------------------------------
    # AbstractStorage interface
    # ------------------------------------------------------------------
 
    def open(
        self,
        uri: str,
        mode: str = "rb",
        *,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> HTTPFile:
        if mode != "rb":
            raise ValueError(f"HTTPStorage only supports mode='rb', got {mode!r}")
 
        self._validate_uri(uri)
        try:
            with timed(f"http_get:{uri}", warn_threshold_seconds=self.timeout):
                response = self.session.get(uri, stream=True, timeout=self.timeout)
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise DataSourceError(
                f"HTTP request returned error status: {uri}",
                context={"uri": uri, "status_code": exc.response.status_code if exc.response else None},
                cause=exc,
            ) from exc
        except requests.RequestException as exc:
            raise DataSourceError(
                f"HTTP request failed: {uri}",
                context={"uri": uri},
                cause=exc,
            ) from exc
 
        logger.info({
            "event": "http_open",
            "uri": uri,
            "status_code": response.status_code,
            "content_length": response.headers.get("Content-Length"),
        })
        return HTTPFile(response, uri, chunk_size=self.chunk_size)
 
    def exists(self, uri: str) -> bool:
        """Send a HEAD request; return ``True`` only on HTTP 200."""
        self._validate_uri(uri)
        try:
            resp = self.session.head(uri, timeout=self.timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False
 
    def list(self, prefix: str, recursive: bool = False) -> List[str]:
        raise NotImplementedError("HTTPStorage does not support directory listing.")
 
    def copy(self, src_uri: str, dst_uri: str, overwrite: bool = False) -> None:
        raise NotImplementedError("HTTPStorage is read-only; copy is not supported.")
 
    def delete(self, uri: str) -> None:
        raise NotImplementedError("HTTPStorage is read-only; delete is not supported.")
 
 
if __name__ == "__main__":
    print("\n=== Running http ===\n")
    printer.status("TEST", "http initialized", "info")

    storage = HTTPStorage()
    printer.status("PASS", "HTTPStorage instantiated", "success")

    # URI validation
    try:
        storage._validate_uri("ftp://example.com/file")
        assert False
    except DataSourceError:
        printer.status("PASS", "_validate_uri rejects ftp://", "success")

    # read(0) short-circuit
    import unittest.mock as mock
    fake_resp = mock.MagicMock(spec=requests.Response)
    fake_resp.iter_content = mock.MagicMock(return_value=iter([b"hello", b" world"]))
    fake_resp.close = mock.MagicMock()

    hf = HTTPFile(response=fake_resp, uri="http://example.com/f", chunk_size=5)
    assert hf.read(0) == b""
    printer.status("PASS", "read(0) returns empty bytes", "success")

    # read(5) exact
    data = hf.read(5)
    assert data == b"hello", f"got {data!r}"
    assert hf.tell() == 5
    printer.status("PASS", "read(5) and tell()", "success")

    # read() drain remainder
    data2 = hf.read()
    assert data2 == b" world"
    printer.status("PASS", "read() drains remainder", "success")

    # seek always raises
    try:
        hf.seek(0)
        assert False
    except NotImplementedError:
        printer.status("PASS", "seek raises NotImplementedError", "success")

    # closed-file guard
    hf.close()
    try:
        hf.read()
        assert False
    except DataSourceError:
        printer.status("PASS", "closed-file guard on read", "success")

    # Unsupported operations - list and delete (single argument)
    for method in ("list", "delete"):
        try:
            getattr(storage, method)("http://x.com/y")
            assert False
        except NotImplementedError:
            pass
    printer.status("PASS", "list/delete raise NotImplementedError", "success")

    # Copy requires two arguments
    try:
        storage.copy("http://x.com/y", "http://x.com/z")
        assert False
    except NotImplementedError:
        printer.status("PASS", "copy raises NotImplementedError", "success")

    print("\n=== Test ran successfully ===\n")