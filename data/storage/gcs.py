from __future__ import annotations
 
from typing import BinaryIO, Dict, Any, List, Optional
from urllib.parse import urlparse
from google.cloud import storage  # type: ignore
from google.cloud.exceptions import NotFound, GoogleCloudError  # type: ignore
 
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_storage import AbstractStorage, StorageFile
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("Google Cloud Storage")
printer = PrettyPrinter()
 
 
class GCSFile(StorageFile):
    """Streaming GCS object reader backed by ``google.cloud.storage.Blob.open()``.
 
    GCS blobs opened with ``blob.open("rb")`` support ``seek`` and ``tell``
    natively, so we expose ``seekable() → True`` and delegate directly.
    The ``closed`` state is tracked on the instance to guard all operations.
    """
 
    def __init__(self, blob: storage.Blob, uri: str) -> None:
        super().__init__()
        self.file_cfg = get_config_section("storage_file")
        self._blob = blob
        self._uri = uri
        self._stream: Optional[BinaryIO] = None
        self._closed: bool = False
 
    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
 
    def _ensure_stream(self) -> BinaryIO:
        if self._stream is None:
            self._stream = self._blob.open("rb")
        assert self._stream is not None
        return self._stream
 
    # ------------------------------------------------------------------
    # StorageFile interface
    # ------------------------------------------------------------------
 
    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise DataSourceError("Read on closed GCS file", context={"uri": self._uri})
        return self._ensure_stream().read(size if size >= 0 else -1)
 
    def seek(self, offset: int, whence: int = 0) -> int:
        if self._closed:
            raise DataSourceError("Seek on closed GCS file", context={"uri": self._uri})
        return self._ensure_stream().seek(offset, whence)
 
    def tell(self) -> int:
        if self._closed:
            raise DataSourceError("Tell on closed GCS file", context={"uri": self._uri})
        if self._stream is None:
            return 0
        return self._stream.tell()
 
    def close(self) -> None:
        if not self._closed:
            if self._stream is not None:
                self._stream.close()
            self._closed = True
 
    def seekable(self) -> bool:
        return True
 
 
class GCSStorage(AbstractStorage):
    """Google Cloud Storage backend.
 
    Configuration is read from ``data_config.yaml → storage.gcs``.
    ``credentials_path`` overrides the ADC (Application Default Credentials)
    discovery chain — useful in CI where a service-account JSON is injected.
    """
 
    def __init__(
        self,
        project: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.storage_cfg = get_config_section("storage")
        self.gcs_cfg: Dict[str, Any] = self.storage_cfg.get("gcs", {})
 
        self.timeout: float = float(self.gcs_cfg.get("timeout", 30.0))
        _project: Optional[str] = project or self.gcs_cfg.get("project")
        _creds_path: Optional[str] = credentials_path or self.gcs_cfg.get("credentials_path")
 
        if _creds_path:
            self.client = storage.Client.from_service_account_json(
                _creds_path, project=_project
            )
        else:
            self.client = storage.Client(project=_project)
 
        logger.info({
            "event": "gcs_storage_init",
            "project": _project,
            "credentials_source": "service_account" if _creds_path else "ADC",
        })
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        """Return ``(bucket, key)`` from a ``gs://bucket/key`` URI."""
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise DataSourceError(
                "Invalid GCS URI scheme — expected gs://",
                context={"uri": uri},
            )
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise DataSourceError(
                "GCS URI must contain both a bucket and a key",
                context={"uri": uri},
            )
        return bucket, key
 
    # ------------------------------------------------------------------
    # AbstractStorage interface
    # ------------------------------------------------------------------
 
    @with_retry(max_attempts=5, retryable_exceptions=(GoogleCloudError, DataSourceError))
    def open(
        self,
        uri: str,
        mode: str = "rb",
        *,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> GCSFile:
        if mode != "rb":
            raise ValueError(f"GCSStorage only supports mode='rb', got {mode!r}")
 
        bucket_name, blob_name = self._parse_uri(uri)
        bucket = self.client.bucket(bucket_name)
 
        with timed(f"gcs_get_blob:{uri}", warn_threshold_seconds=self.timeout):
            blob = bucket.get_blob(blob_name, timeout=self.timeout)
 
        if blob is None:
            raise DataSourceError(
                f"GCS object not found: {uri}",
                context={"uri": uri, "bucket": bucket_name, "key": blob_name},
            )
 
        logger.info({"event": "gcs_open", "uri": uri, "size_bytes": blob.size})
        return GCSFile(blob, uri)
 
    def exists(self, uri: str) -> bool:
        bucket_name, blob_name = self._parse_uri(uri)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name, timeout=self.timeout)
        return blob is not None
 
    def list(self, prefix: str, recursive: bool = False) -> List[str]:
        bucket_name, blob_prefix = self._parse_uri(prefix)
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=blob_prefix, timeout=self.timeout)
        uris: List[str] = []
        for blob in blobs:
            relative = blob.name[len(blob_prefix):]
            if not recursive and "/" in relative:
                continue
            uris.append(f"gs://{bucket_name}/{blob.name}")
        return sorted(uris)
 
    def copy(self, src_uri: str, dst_uri: str, overwrite: bool = False) -> None:
        src_bucket_name, src_key = self._parse_uri(src_uri)
        dst_bucket_name, dst_key = self._parse_uri(dst_uri)
 
        if not overwrite and self.exists(dst_uri):
            raise DataSourceError(
                f"Destination already exists: {dst_uri}",
                context={"dst": dst_uri},
            )
 
        src_bucket = self.client.bucket(src_bucket_name)
        src_blob = src_bucket.get_blob(src_key, timeout=self.timeout)
        if src_blob is None:
            raise DataSourceError(
                f"Source GCS object not found: {src_uri}",
                context={"src": src_uri},
            )
 
        dst_bucket = self.client.bucket(dst_bucket_name)
        src_bucket.copy_blob(src_blob, dst_bucket, dst_key, timeout=self.timeout)
        logger.info({"event": "gcs_copy", "src": src_uri, "dst": dst_uri})
 
    def delete(self, uri: str) -> None:
        bucket_name, blob_name = self._parse_uri(uri)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name, timeout=self.timeout)
        if blob is not None:
            blob.delete(timeout=self.timeout)
            logger.info({"event": "gcs_delete", "uri": uri})
 
 
if __name__ == "__main__":
    print("\n=== Running gcs ===\n")
    printer.status("TEST", "gcs initialized", "info")
 
    # URI parsing — valid
    bucket, key = GCSStorage._parse_uri("gs://my-bucket/data/file.parquet")
    assert bucket == "my-bucket" and key == "data/file.parquet"
    printer.status("PASS", "_parse_uri valid URI", "success")
 
    # URI parsing — rejects bad schemes
    for bad in ["s3://bucket/key", "gs://", "gs://bucket-only"]:
        try:
            GCSStorage._parse_uri(bad)
            assert False, f"expected DataSourceError for {bad!r}"
        except DataSourceError:
            pass
    printer.status("PASS", "_parse_uri rejects invalid URIs", "success")
 
    # GCSFile closed-file guards (no network needed)
    import unittest.mock as mock
 
    fake_blob = mock.MagicMock(spec=storage.Blob)
    fake_blob.size = 42
    gf = GCSFile(blob=fake_blob, uri="gs://b/k")
    gf._closed = True
 
    for method, args in [("read", ()), ("seek", (0,)), ("tell", ())]:
        try:
            getattr(gf, method)(*args)
            assert False, f"{method} should raise on closed file"
        except DataSourceError:
            pass
    printer.status("PASS", "GCSFile closed-file guards", "success")
 
    # GCSFile seekable flag
    gf2 = GCSFile(blob=fake_blob, uri="gs://b/k2")
    assert gf2.seekable() is True
    printer.status("PASS", "GCSFile.seekable() is True", "success")
 
    # GCSFile tell() returns 0 when stream not yet opened
    assert gf2.tell() == 0
    printer.status("PASS", "GCSFile.tell() pre-open returns 0", "success")
    gf2.close()
 
    print("\n=== Test ran successfully ===\n")