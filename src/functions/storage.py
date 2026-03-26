"""Pluggable file storage abstraction with local filesystem and S3 backends.

Provides consistent interface for uploading, downloading, and deleting files.
"""

from __future__ import annotations

import os
import uuid
import shutil

from pathlib import Path
from typing import Optional, BinaryIO, Dict, Any, Union

from .utils.config_loader import get_config_section
from .utils.functions_error import StorageError
from .utils.storage_backend import StorageBackend
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Storage")
printer = PrettyPrinter


class LocalStorage(StorageBackend):
    """Store files on the local filesystem."""
    def __init__(self, base_path: str, base_url: Optional[str] = None):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip('/') if base_url else None

    def _resolve_path(self, path: str) -> Path:
        full = (self.base_path / path).resolve()
        # Security: ensure path stays inside base_path
        if self.base_path not in full.parents and full != self.base_path:
            raise StorageError("Path traversal detected")
        return full

    def upload(self, file_obj: BinaryIO, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        full = self._resolve_path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(full, 'wb') as f:
                shutil.copyfileobj(file_obj, f)
            logger.info(f"Uploaded to {full}")
            return str(full)
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise StorageError(f"Local storage upload failed: {e}")

    def download(self, path: str) -> bytes:
        full = self._resolve_path(path)
        try:
            with open(full, 'rb') as f:
                return f.read()
        except Exception as e:
            raise StorageError(f"Download failed: {e}")

    def delete(self, path: str) -> None:
        full = self._resolve_path(path)
        try:
            if full.is_file():
                full.unlink()
            elif full.is_dir():
                shutil.rmtree(full)
            else:
                raise StorageError(f"Path does not exist: {path}")
            logger.info(f"Deleted {full}")
        except Exception as e:
            raise StorageError(f"Delete failed: {e}")

    def get_url(self, path: str) -> str:
        if self.base_url:
            return f"{self.base_url}/{path}"
        # Fallback to file:// URL
        return f"file://{self._resolve_path(path)}"


# Optional S3 backend (if boto3 is installed)
class S3Storage(StorageBackend):
    """Amazon S3 storage backend."""
    def __init__(self, bucket: str, region: str = None, access_key: str = None, secret_key: str = None,
                 endpoint_url: str = None, public_url_prefix: str = None):
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("boto3 is required for S3 storage")

        self.bucket = bucket
        self.client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
        )
        self.public_url_prefix = public_url_prefix

    def upload(self, file_obj: BinaryIO, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = {str(k): str(v) for k, v in metadata.items()}
            self.client.upload_fileobj(file_obj, self.bucket, path, ExtraArgs=extra_args)
            logger.info(f"Uploaded to s3://{self.bucket}/{path}")
            return path
        except Exception as e:
            raise StorageError(f"S3 upload failed: {e}")

    def download(self, path: str) -> bytes:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=path)
            return obj['Body'].read()
        except Exception as e:
            raise StorageError(f"S3 download failed: {e}")

    def delete(self, path: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=path)
            logger.info(f"Deleted s3://{self.bucket}/{path}")
        except Exception as e:
            raise StorageError(f"S3 delete failed: {e}")

    def get_url(self, path: str) -> str:
        if self.public_url_prefix:
            return f"{self.public_url_prefix}/{path}"
        # Fallback to presigned URL (temporary)
        try:
            return self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': path},
                ExpiresIn=3600
            )
        except Exception:
            raise StorageError("Cannot generate URL; public prefix not set")


class Storage:
    """
    Facade for storage backends. Use from_config to create instance.
    """
    def __init__(self, backend: StorageBackend, generate_unique_filename: bool = True):
        self.backend = backend
        self.generate_unique_filename = generate_unique_filename

    def _unique_filename(self, original_name: str) -> str:
        """Generate a unique filename while preserving extension."""
        ext = Path(original_name).suffix
        return f"{uuid.uuid4().hex}{ext}"

    def upload(self, file_obj: BinaryIO, filename: str, subpath: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload a file. If `generate_unique_filename` is True, a UUID is inserted.
        The final path is `subpath/final_filename`.
        """
        if self.generate_unique_filename:
            final_name = self._unique_filename(filename)
        else:
            final_name = filename
        if subpath:
            path = f"{subpath}/{final_name}"
        else:
            path = final_name
        return self.backend.upload(file_obj, path, metadata)

    def download(self, path: str) -> bytes:
        return self.backend.download(path)

    def delete(self, path: str) -> None:
        self.backend.delete(path)

    def get_url(self, path: str) -> str:
        return self.backend.get_url(path)

    @classmethod
    def from_config(cls):
        """Create Storage from configuration section 'storage'."""
        config = get_config_section('storage')
        backend_type = config.get('backend', 'local')

        if backend_type == 'local':
            backend = LocalStorage(
                base_path=config.get('base_path', 'data/storage'),
                base_url=config.get('base_url')
            )
        elif backend_type == 's3':
            backend = S3Storage(
                bucket=config['bucket'],
                region=config.get('region'),
                access_key=config.get('access_key'),
                secret_key=config.get('secret_key'),
                endpoint_url=config.get('endpoint_url'),
                public_url_prefix=config.get('public_url_prefix')
            )
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")

        generate_unique = config.get('generate_unique_filename', True)
        return cls(backend, generate_unique)
    

if __name__ == "__main__":
    print("\n=== Running Storage ===\n")
    printer.status("TEST", "Starting Storage tests", "info")
    backend = LocalStorage(base_path="data/storage", base_url=None)

    storage = Storage(
        backend=backend,
        generate_unique_filename=True
    )
    print(storage)
    print("\n=== Successfully ran the Storage ===\n")