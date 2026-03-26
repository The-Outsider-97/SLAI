
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Dict, Optional


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    @abstractmethod
    def upload(self, file_obj: BinaryIO, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upload a file and return its identifier or full path."""
        pass

    @abstractmethod
    def download(self, path: str) -> bytes:
        """Download file content as bytes."""
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file."""
        pass

    @abstractmethod
    def get_url(self, path: str) -> str:
        """Return a publicly accessible URL (if available)."""
        pass
