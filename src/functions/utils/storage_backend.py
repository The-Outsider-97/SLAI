"""Abstract contract for pluggable storage backends.

The backend interface is intentionally small and backend-neutral so callers can
switch between local filesystem, S3, or other object stores without changing
application code.

Contract notes:
- Paths are backend-neutral object keys, for example ``docs/report.pdf``.
- Implementations should not return absolute local filesystem paths from
  ``upload(...)``; they should return the normalized object key that callers can
  later pass to ``download(...)``, ``delete(...)``, and ``get_url(...)``.
- ``delete(...)`` is a file/object delete operation. Implementations should not
  recursively delete directories or prefixes through this API.
- Methods should raise ``StorageError`` or one of its subclasses on failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Dict, Optional

__all__ = ["StorageBackend"]


class StorageBackend(ABC):
    """Abstract base class for backend-neutral storage implementations.

    Concrete implementations are responsible for validating object keys,
    normalizing metadata, and translating backend-specific failures into the
    storage exception hierarchy used by the package.
    """

    @abstractmethod
    def upload(
        self,
        file_obj: BinaryIO,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload a file-like object's content to the given object key.

        Args:
            file_obj: Binary file-like object positioned at the content to read.
            path: Backend-neutral destination object key, such as
                ``"images/avatar.png"``.
            metadata: Optional key-value metadata associated with the object.

        Returns:
            The stored backend-neutral object key. Implementations should return
            the normalized key, not an absolute filesystem path or provider-
            specific URI, so the return value can be reused with other storage
            methods.
        """
        raise NotImplementedError

    @abstractmethod
    def download(self, path: str) -> bytes:
        """Download and return the bytes stored at the given object key.

        Args:
            path: Backend-neutral object key previously returned by
                :meth:`upload`.

        Returns:
            The complete object content as bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a single stored object identified by the given key.

        Args:
            path: Backend-neutral object key to remove.

        Notes:
            This method is defined as a file/object deletion API. Backends
            should not treat directory trees or object prefixes as valid delete
            targets for this contract.
        """
        raise NotImplementedError

    @abstractmethod
    def get_url(self, path: str) -> str:
        """Return a retrievable URL for the given object key.

        The returned URL may be public, internal, or time-limited depending on
        the backend.

        Args:
            path: Backend-neutral object key.

        Returns:
            A URL string that can be used to retrieve the object.
        """
        raise NotImplementedError
