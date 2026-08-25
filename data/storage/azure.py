from __future__ import annotations

import os
 
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient, BlobClient  # pyright: ignore[reportMissingImports]
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError  # pyright: ignore[reportMissingImports]

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_storage import AbstractStorage, StorageFile
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Azure File")
printer = PrettyPrinter()


class AzureFile(StorageFile):
    """Streaming Azure Blob reader.
 
    Azure ``download_blob()`` returns a ``StorageStreamDownloader`` whose
    ``readall()`` / ``readinto()`` / ``chunks()`` methods stream data without
    buffering the entire blob in memory.  We wrap the downloader in a minimal
    ``read``-only interface; seeking is not natively supported for streaming
    blobs and is therefore restricted to offset-0 by re-downloading.
    """
 
    def __init__(self, blob_client: BlobClient, uri: str) -> None:
        super().__init__()
        self.file_cfg = get_config_section("storage_file")
        self._blob_client = blob_client
        self._uri = uri
        self._downloader = None          # azure StorageStreamDownloader
        self._pos: int = 0
        self._closed: bool = False
 
    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
 
    def _ensure_downloader(self):
        if self._downloader is None:
            self._downloader = self._blob_client.download_blob()
        return self._downloader
 
    # ------------------------------------------------------------------
    # StorageFile interface
    # ------------------------------------------------------------------
 
    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise DataSourceError("Read on closed Azure file", context={"uri": self._uri})
        dl = self._ensure_downloader()
        data = dl.read(size) if size >= 0 else dl.readall()
        self._pos += len(data)
        return data
 
    def seek(self, offset: int, whence: int = 0) -> int:
        """Only seek-to-start (offset=0, whence=0) is supported.
 
        A seek to the beginning re-issues the ``download_blob()`` call.  Any
        other combination raises ``NotImplementedError`` because Azure streaming
        downloads do not expose a random-access interface without a new HTTP
        request.
        """
        if self._closed:
            raise DataSourceError("Seek on closed Azure file", context={"uri": self._uri})
        if offset == 0 and whence == 0:
            if self._downloader is not None:
                # There is no explicit close on StorageStreamDownloader; let GC handle it.
                self._downloader = None
            self._pos = 0
            return 0
        raise NotImplementedError(
            "AzureFile only supports seek(0, 0); re-open the blob for arbitrary access."
        )
 
    def tell(self) -> int:
        if self._closed:
            raise DataSourceError("Tell on closed Azure file", context={"uri": self._uri})
        return self._pos
 
    def close(self) -> None:
        if not self._closed:
            self._downloader = None
            self._closed = True
 
 
class AzureStorage(AbstractStorage):
    """Azure Blob Storage backend.
 
    Credentials are resolved in priority order:
    1. Explicit ``connection_string`` argument.
    2. ``connection_string_env`` — name of an environment variable that holds
       the connection string (configured in ``data_config.yaml``).
    3. Explicit ``account_url + credential`` arguments.
 
    Configuration is read from ``data_config.yaml → storage.azure``.
    """
 
    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.storage_cfg = get_config_section("storage")
        self.azure_cfg: Dict[str, Any] = self.storage_cfg.get("azure", {})
        self.timeout: float = float(self.azure_cfg.get("timeout", 30.0))
 
        # Resolve connection string — explicit arg beats env-var from config.
        _conn_str: Optional[str] = connection_string
        if not _conn_str:
            env_var: str = self.azure_cfg.get("connection_string_env", "")
            if env_var:
                _conn_str = os.environ.get(env_var)
 
        if _conn_str:
            self.client = BlobServiceClient.from_connection_string(_conn_str)
            logger.info({"event": "azure_storage_init", "auth": "connection_string"})
        elif account_url and credential:
            self.client = BlobServiceClient(account_url=account_url, credential=credential)
            logger.info({"event": "azure_storage_init", "auth": "account_url+credential"})
        else:
            raise DataConfigError(
                "AzureStorage requires a connection string (direct or via env var) "
                "or account_url + credential.",
                context={"azure_cfg_keys": list(self.azure_cfg.keys())},
            )
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        """Return ``(container, blob_name)`` from an ``azure://container/blob`` URI."""
        parsed = urlparse(uri)
        if parsed.scheme != "azure":
            raise DataSourceError(
                "Invalid Azure URI scheme — expected azure://",
                context={"uri": uri},
            )
        container = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        if not container or not blob_name:
            raise DataSourceError(
                "Azure URI must contain both a container and a blob name",
                context={"uri": uri},
            )
        return container, blob_name
 
    # ------------------------------------------------------------------
    # AbstractStorage interface
    # ------------------------------------------------------------------
 
    @with_retry(max_attempts=5, retryable_exceptions=(HttpResponseError, DataSourceError))
    def open(
        self,
        uri: str,
        mode: str = "rb",
        *,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> AzureFile:
        if mode != "rb":
            raise ValueError(f"AzureStorage only supports mode='rb', got {mode!r}")
 
        container, blob_name = self._parse_uri(uri)
        container_client = self.client.get_container_client(container)
 
        if not container_client.exists(timeout=self.timeout):
            raise DataSourceError(
                f"Azure container not found: {container}",
                context={"uri": uri, "container": container},
            )
 
        blob_client = container_client.get_blob_client(blob_name)
        if not blob_client.exists(timeout=self.timeout):
            raise DataSourceError(
                f"Azure blob not found: {uri}",
                context={"uri": uri, "container": container, "blob": blob_name},
            )
 
        logger.info({"event": "azure_open", "uri": uri})
        return AzureFile(blob_client, uri)
 
    def exists(self, uri: str) -> bool:
        container, blob_name = self._parse_uri(uri)
        container_client = self.client.get_container_client(container)
        if not container_client.exists(timeout=self.timeout):
            return False
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.exists(timeout=self.timeout)
 
    def list(self, prefix: str, recursive: bool = False) -> List[str]:
        container, blob_prefix = self._parse_uri(prefix)
        container_client = self.client.get_container_client(container)
        uris: List[str] = []
        for blob in container_client.list_blobs(name_starts_with=blob_prefix):
            relative = blob.name[len(blob_prefix):]
            if not recursive and "/" in relative:
                continue
            uris.append(f"azure://{container}/{blob.name}")
        return sorted(uris)
 
    def copy(self, src_uri: str, dst_uri: str, overwrite: bool = False) -> None:
        src_container, src_blob_name = self._parse_uri(src_uri)
        dst_container, dst_blob_name = self._parse_uri(dst_uri)
 
        if not overwrite and self.exists(dst_uri):
            raise DataSourceError(
                f"Destination already exists: {dst_uri}",
                context={"dst": dst_uri},
            )
 
        src_blob_client = (
            self.client.get_container_client(src_container)
            .get_blob_client(src_blob_name)
        )
        dst_blob_client = (
            self.client.get_container_client(dst_container)
            .get_blob_client(dst_blob_name)
        )
        dst_blob_client.start_copy_from_url(src_blob_client.url, requires_sync=True)
        logger.info({"event": "azure_copy", "src": src_uri, "dst": dst_uri})
 
    def delete(self, uri: str) -> None:
        container, blob_name = self._parse_uri(uri)
        blob_client = (
            self.client.get_container_client(container)
            .get_blob_client(blob_name)
        )
        if blob_client.exists(timeout=self.timeout):
            blob_client.delete_blob()
            logger.info({"event": "azure_delete", "uri": uri})

 
if __name__ == "__main__":
    print("\n=== Running azure ===\n")
    printer.status("TEST", "azure initialized", "info")
 
    # URI parsing — valid
    container, blob = AzureStorage._parse_uri("azure://my-container/path/to/file.bin")
    assert container == "my-container" and blob == "path/to/file.bin"
    printer.status("PASS", "_parse_uri valid URI", "success")
 
    # URI parsing — invalid schemes
    for bad in ["s3://bucket/key", "azure://", "azure://container-only"]:
        try:
            AzureStorage._parse_uri(bad)
            assert False, f"expected DataSourceError for {bad!r}"
        except DataSourceError:
            pass
    printer.status("PASS", "_parse_uri rejects invalid URIs", "success")
 
    # AzureFile closed-file guards (no network needed)
    import unittest.mock as mock
    from azure.storage.blob import BlobClient  # pyright: ignore[reportMissingImports]
 
    fake_client = mock.MagicMock(spec=BlobClient)
    af = AzureFile(blob_client=fake_client, uri="azure://c/b")
    af._closed = True
 
    for method, args in [("read", ()), ("seek", (0,)), ("tell", ())]:
        try:
            getattr(af, method)(*args)
            assert False, f"{method} should raise on closed file"
        except DataSourceError:
            pass
    printer.status("PASS", "AzureFile closed-file guards", "success")
 
    # AzureFile seek(0,0) resets downloader; arbitrary seek raises
    af2 = AzureFile(blob_client=fake_client, uri="azure://c/b2")
    af2._pos = 100
    result = af2.seek(0, 0)
    assert result == 0 and af2._pos == 0
    printer.status("PASS", "AzureFile seek(0,0) resets position", "success")
 
    try:
        af2.seek(5)
        assert False
    except NotImplementedError:
        printer.status("PASS", "AzureFile arbitrary seek raises NotImplementedError", "success")
    af2.close()
 
    # Missing credentials must raise DataConfigError
    try:
        AzureStorage()
        assert False
    except DataConfigError:
        printer.status("PASS", "AzureStorage rejects missing credentials", "success")
 
    print("\n=== Test ran successfully ===\n")