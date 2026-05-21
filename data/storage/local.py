from __future__ import annotations
 
import shutil
 
from pathlib import Path
from typing import BinaryIO, Dict, Any, List, Optional
 
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_storage import AbstractStorage, StorageFile
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("Local Data Storage")
printer = PrettyPrinter()
 
 
def _strip_file_scheme(uri: str) -> str:
    """Remove an optional ``file://`` prefix and return the raw path string."""
    return uri[7:] if uri.startswith("file://") else uri
 
 
class LocalFile(StorageFile):
    """Thin wrapper around a regular binary file object.
 
    Inherits context-manager support from ``StorageFile``; delegates all I/O
    directly to the underlying ``BinaryIO`` so that ``seek`` and ``tell`` work
    without any additional buffering layer.
    """
 
    def __init__(self, fileobj: BinaryIO, path: Path) -> None:
        super().__init__()
        self._fileobj = fileobj
        self._path = path
 
    # ------------------------------------------------------------------
    # StorageFile interface
    # ------------------------------------------------------------------
 
    def read(self, size: int = -1) -> bytes:
        try:
            return self._fileobj.read(size)
        except OSError as exc:
            raise DataSourceError(
                f"Read error on local file: {self._path}",
                context={"path": str(self._path)},
                cause=exc,
            ) from exc
 
    def seek(self, offset: int, whence: int = 0) -> int:
        try:
            return self._fileobj.seek(offset, whence)
        except OSError as exc:
            raise DataSourceError(
                f"Seek error on local file: {self._path}",
                context={"path": str(self._path), "offset": offset, "whence": whence},
                cause=exc,
            ) from exc
 
    def tell(self) -> int:
        return self._fileobj.tell()
 
    def close(self) -> None:
        self._fileobj.close()
 
    def seekable(self) -> bool:
        return True
 
 
class LocalStorage(AbstractStorage):
    """Local-filesystem storage backend.
 
    Accepts both raw POSIX/Windows paths and ``file://``-prefixed URIs.
    All path resolution goes through ``resolve_path`` (from ``data_helpers``)
    to prevent path-traversal attacks and reject symlinks.
    """
 
    def __init__(self) -> None:
        super().__init__()
        self.storage_cfg = get_config_section("storage")
        self.local_cfg: Dict[str, Any] = self.storage_cfg.get("local", {})
        self.allow_symlinks: bool = bool(self.local_cfg.get("allow_symlinks", False))
        logger.info({"event": "local_storage_init", "allow_symlinks": self.allow_symlinks})
 
    # ------------------------------------------------------------------
    # AbstractStorage interface
    # ------------------------------------------------------------------
    def open(self, uri: str, mode: str = "rb", *,
             retry_config: Optional[Dict[str, Any]] = None) -> LocalFile:
        if mode != "rb":
            raise ValueError(f"LocalStorage only supports mode='rb', got {mode!r}")
 
        path = resolve_path(
            _strip_file_scheme(uri),
            must_exist=True,
            allow_symlinks=self.allow_symlinks,
        )
 
        try:
            fileobj: BinaryIO = open(path, "rb")
        except OSError as exc:
            raise DataSourceError(
                f"Cannot open local file: {path}",
                context={"uri": uri, "path": str(path)},
                cause=exc,
            ) from exc
 
        logger.debug({"event": "local_open", "path": str(path)})
        return LocalFile(fileobj, path)
 
    def exists(self, uri: str) -> bool:
        return Path(_strip_file_scheme(uri)).exists()
 
    def list(self, prefix: str, recursive: bool = False) -> List[str]:
        root = Path(_strip_file_scheme(prefix))
        if not root.exists() or not root.is_dir():
            return []
        pattern = "**/*" if recursive else "*"
        return sorted(
            f"file://{p.resolve()}"
            for p in root.glob(pattern)
            if p.is_file()
        )
 
    def copy(self, src_uri: str, dst_uri: str, overwrite: bool = False) -> None:
        src_path = Path(_strip_file_scheme(src_uri))
        dst_path = Path(_strip_file_scheme(dst_uri))
 
        if not src_path.exists():
            raise DataSourceError(
                f"Source file not found: {src_uri}",
                context={"src": str(src_path)},
            )
        if not overwrite and dst_path.exists():
            raise DataSourceError(
                f"Destination already exists: {dst_uri}",
                context={"dst": str(dst_path)},
            )
 
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        logger.info({"event": "local_copy", "src": str(src_path), "dst": str(dst_path)})
 
    def delete(self, uri: str) -> None:
        path = Path(_strip_file_scheme(uri))
        if path.exists():
            path.unlink()
            logger.info({"event": "local_delete", "path": str(path)})
 
 
if __name__ == "__main__":
    import tempfile
 
    print("\n=== Running local ===\n")
    printer.status("TEST", "local initialized", "info")
 
    storage = LocalStorage()
    printer.status("PASS", "LocalStorage instantiated", "success")
 
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "hello.bin"
        src.write_bytes(b"hello storage")
 
        # open + read
        with storage.open(str(src)) as fh:
            data = fh.read()
        assert data == b"hello storage", f"unexpected data: {data!r}"
        printer.status("PASS", "open/read", "success")
 
        # seek + tell
        with storage.open(str(src)) as fh:
            fh.seek(6)
            assert fh.tell() == 6
            assert fh.read() == b"storage"
        printer.status("PASS", "seek/tell", "success")
 
        # exists
        assert storage.exists(str(src))
        assert not storage.exists(str(Path(tmpdir) / "missing.bin"))
        printer.status("PASS", "exists", "success")
 
        # list
        (Path(tmpdir) / "sub").mkdir()
        (Path(tmpdir) / "sub" / "b.bin").write_bytes(b"x")
        flat = storage.list(tmpdir, recursive=False)
        deep = storage.list(tmpdir, recursive=True)
        assert any("hello.bin" in u for u in flat)
        assert any("b.bin" in u for u in deep)
        printer.status("PASS", "list flat + recursive", "success")
 
        # copy
        dst = Path(tmpdir) / "copy.bin"
        storage.copy(str(src), str(dst))
        assert dst.read_bytes() == b"hello storage"
        printer.status("PASS", "copy", "success")
 
        # copy – overwrite guard
        try:
            storage.copy(str(src), str(dst), overwrite=False)
            assert False, "expected DataSourceError"
        except DataSourceError:
            printer.status("PASS", "copy overwrite guard", "success")
 
        # delete
        storage.delete(str(dst))
        assert not dst.exists()
        printer.status("PASS", "delete", "success")
 
        # invalid mode
        try:
            storage.open(str(src), mode="wb")
            assert False
        except ValueError:
            printer.status("PASS", "invalid mode rejected", "success")
 
    print("\n=== Test ran successfully ===\n")