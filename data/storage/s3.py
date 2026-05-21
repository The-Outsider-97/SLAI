from __future__ import annotations
 
import boto3  # type: ignore
 
from typing import BinaryIO, Dict, Any, List, Optional
from botocore.config import Config  # type: ignore
from botocore.exceptions import ClientError, BotoCoreError  # type: ignore
 
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_storage import AbstractStorage, StorageFile
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("S3 Storage")
printer = PrettyPrinter()
 
 
class S3File(StorageFile):
    """Thin wrapper around a botocore ``StreamingBody``.
 
    ``StreamingBody`` exposes ``.read()`` but not ``seek``/``tell``.
    We track the position manually so that ``tell`` is always accurate.
    ``seek`` is intentionally restricted: rewinding to offset 0 re-opens the
    object via a fresh ``GetObject`` call; arbitrary seeks are unsupported
    because S3 range requests require a new HTTP connection.
    """
 
    def __init__(self, body: BinaryIO, size: int, uri: str) -> None:
        super().__init__()
        self.file_cfg = get_config_section("storage_file")
        self._body = body
        self._size = size
        self._uri = uri
        self._pos: int = 0
        self._closed: bool = False
 
    # ------------------------------------------------------------------
    # StorageFile interface
    # ------------------------------------------------------------------
    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise DataSourceError("Read on closed S3 file", context={"uri": self._uri})
        # Use -1 to indicate "read all" (consistent with io.RawIOBase)
        chunk_size = size if size >= 0 else -1
        data = self._body.read(chunk_size)
        self._pos += len(data)
        return data
 
    def seek(self, offset: int, whence: int = 0) -> int:
        # Only seek-to-start is meaningful without a Range re-request.
        if offset == 0 and whence == 0:
            raise NotImplementedError(
                "S3File.seek(0) is not supported inline; re-open the object via S3Storage.open()."
            )
        raise NotImplementedError("Arbitrary seek is not supported for S3 streaming objects.")
 
    def tell(self) -> int:
        if self._closed:
            raise DataSourceError(
                "Tell on closed S3 file", context={"uri": self._uri}
            )
        return self._pos
 
    def close(self) -> None:
        if not self._closed:
            self._body.close()
            self._closed = True
 
    @property
    def size(self) -> int:
        """Content-Length reported by S3 at open time."""
        return self._size
 
 
class S3Storage(AbstractStorage):
    """AWS S3 backend with adaptive retries, configurable timeouts, and IAM support.
 
    Configuration is read from ``data_config.yaml → storage.s3``.
    Constructor arguments override config values to support per-instance
    customisation (e.g. localstack in tests).
    """
 
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.storage_cfg = get_config_section("storage")
        self.s3_cfg: Dict[str, Any] = self.storage_cfg.get("s3", {})
        retry_cfg: Dict[str, Any] = self.storage_cfg.get("retry", {})
 
        self.timeout: float = float(self.s3_cfg.get("timeout", 30.0))
        self.max_attempts: int = int(retry_cfg.get("max_attempts", 5))
        _region: str = region or self.s3_cfg.get("region", "us-east-1")
 
        boto_config = Config(
            region_name=_region,
            retries={"max_attempts": self.max_attempts, "mode": "adaptive"},
            connect_timeout=self.timeout,
            read_timeout=self.timeout,
        )
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url, config=boto_config)
        logger.info({
            "event": "s3_storage_init",
            "region": _region,
            "max_attempts": self.max_attempts,
            "timeout": self.timeout,
            "endpoint_url": endpoint_url,
        })
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        """Return ``(bucket, key)`` from an ``s3://bucket/key`` URI."""
        if not uri.startswith("s3://"):
            raise DataSourceError(
                "Invalid S3 URI scheme — expected s3://",
                context={"uri": uri},
            )
        remainder = uri[5:]
        parts = remainder.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        if not bucket or not key:
            raise DataSourceError(
                "S3 URI must contain both a bucket and a key",
                context={"uri": uri},
            )
        return bucket, key
 
    # ------------------------------------------------------------------
    # AbstractStorage interface
    # ------------------------------------------------------------------
 
    @with_retry(max_attempts=5, retryable_exceptions=(ClientError, BotoCoreError, DataSourceError))
    def open(
        self,
        uri: str,
        mode: str = "rb",
        *,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> S3File:
        if mode != "rb":
            raise ValueError(f"S3Storage only supports mode='rb', got {mode!r}")
 
        bucket, key = self._parse_uri(uri)
        try:
            with timed(f"s3_get_object:{uri}", warn_threshold_seconds=self.timeout):
                response = self.s3.get_object(Bucket=bucket, Key=key)
            body: BinaryIO = response["Body"]
            size: int = response.get("ContentLength", 0)
            logger.info({"event": "s3_open", "uri": uri, "size_bytes": size})
            return S3File(body, size, uri)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "NoSuchKey":
                raise DataSourceError(
                    f"S3 object not found: {uri}",
                    context={"uri": uri, "bucket": bucket, "key": key},
                    cause=exc,
                ) from exc
            raise DataSourceError(
                f"S3 GetObject failed: {uri}",
                context={"uri": uri, "error_code": code},
                cause=exc,
            ) from exc
 
    def exists(self, uri: str) -> bool:
        bucket, key = self._parse_uri(uri)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise DataSourceError(
                f"S3 existence check failed: {uri}",
                context={"uri": uri},
                cause=exc,
            ) from exc
 
    def list(self, prefix: str, recursive: bool = False) -> List[str]:
        bucket, key_prefix = self._parse_uri(prefix)
        uris: List[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not recursive and "/" in key[len(key_prefix):]:
                    continue
                uris.append(f"s3://{bucket}/{key}")
        return sorted(uris)
 
    def copy(self, src_uri: str, dst_uri: str, overwrite: bool = False) -> None:
        src_bucket, src_key = self._parse_uri(src_uri)
        dst_bucket, dst_key = self._parse_uri(dst_uri)
 
        if not overwrite and self.exists(dst_uri):
            raise DataSourceError(
                f"Destination already exists: {dst_uri}",
                context={"dst": dst_uri},
            )
        try:
            self.s3.copy_object(
                CopySource={"Bucket": src_bucket, "Key": src_key},
                Bucket=dst_bucket,
                Key=dst_key,
            )
            logger.info({"event": "s3_copy", "src": src_uri, "dst": dst_uri})
        except ClientError as exc:
            raise DataSourceError(
                f"S3 copy failed: {src_uri} → {dst_uri}",
                context={"src": src_uri, "dst": dst_uri},
                cause=exc,
            ) from exc
 
    def delete(self, uri: str) -> None:
        bucket, key = self._parse_uri(uri)
        try:
            self.s3.delete_object(Bucket=bucket, Key=key)
            logger.info({"event": "s3_delete", "uri": uri})
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "NoSuchKey":
                raise DataSourceError(
                    f"S3 delete failed: {uri}",
                    context={"uri": uri},
                    cause=exc,
                ) from exc
 
 
if __name__ == "__main__":
    print("\n=== Running s3 ===\n")
    printer.status("TEST", "s3 initialized", "info")
 
    # URI parsing
    bucket, key = S3Storage._parse_uri("s3://my-bucket/path/to/file.parquet")
    assert bucket == "my-bucket" and key == "path/to/file.parquet"
    printer.status("PASS", "_parse_uri valid URI", "success")
 
    for bad in ["gs://bucket/key", "s3://", "s3://bucket-only"]:
        try:
            S3Storage._parse_uri(bad)
            assert False, f"expected DataSourceError for {bad!r}"
        except DataSourceError:
            pass
    printer.status("PASS", "_parse_uri rejects invalid URIs", "success")
 
    # S3File read/tell/close with a mock body
    import io
    body = io.BytesIO(b"hello s3")
    f = S3File(body, size=8, uri="s3://b/k")
    assert f.tell() == 0
    assert f.read(5) == b"hello"
    assert f.tell() == 5
    assert f.read() == b" s3"
    assert f.tell() == 8
    f.close()
    printer.status("PASS", "S3File read/tell/close", "success")
 
    # Closed-file guard
    try:
        f.read()
        assert False
    except DataSourceError:
        printer.status("PASS", "S3File closed-file guard", "success")
 
    # Seek raises NotImplementedError
    f2 = S3File(io.BytesIO(b"x"), size=1, uri="s3://b/k2")
    try:
        f2.seek(0)
        assert False
    except NotImplementedError:
        printer.status("PASS", "S3File seek raises NotImplementedError", "success")
    f2.close()
 
    print("\n=== Test ran successfully ===\n")