"""Right-of-access / portability providers for SLAI's real stores (GDPR Art. 15, 20).

Mirror of ``deletion_adapters``: the same linkage convention (``subject_tag``), the same stores, the same
known limits. A provider exports only what it can attribute to the (tenant, subject); it never guesses.

``portable`` (Art. 20(1): data the subject *provided*, processed on consent/contract, by automated means) cannot be
derived from a store, so it defaults to False and is switched on only by an explicit deployment decision.
"""
from __future__ import annotations

import base64
import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .deletion_adapters import subject_tag


class SharedMemoryRightsProvider:
    def __init__(self, memory: Any, *, name: str = "shared_memory", portable: bool = False) -> None:
        self.name, self.portable, self._m = name, portable, memory

    def export(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        items = [{"key": i["key"], "value": copy.deepcopy(i["value"]), "tags": list(i.get("tags", []))}
                 for i in (self._m.get_by_tag(subject_tag(tenant_id, subject_id)) or [])]
        return {"items": sorted(items, key=lambda i: str(i["key"]))}


class SecureMemoryRightsProvider:
    def __init__(self, memory: Any, *, name: str = "secure_memory", portable: bool = False) -> None:
        self.name, self.portable, self._m = name, portable, memory

    def export(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        ctx = self._m.internal_context("subject_access", principal="governance_access", tenant_id=tenant_id)
        out: List[Dict[str, Any]] = []
        for entry_id in sorted(self._m.tag_index.get(subject_tag(tenant_id, subject_id), set())):
            entry = self._m.get(entry_id, ctx)  # SecureMemory enforces tenant isolation itself: another tenant's entry returns None
            if entry is None:
                continue
            out.append({"entry_id": entry_id, "data": copy.deepcopy(entry.get("data")), "created_at": (entry.get("meta") or {}).get("created_at")})
        return {"entries": out}


class PrivacyMemoryRightsProvider:
    """Exports from *all live* PrivacyMemory instances (each privacy component owns one)."""

    def __init__(self, *, tenant_id: str, name: str = "privacy_memory", portable: bool = False) -> None:
        self.name, self.portable, self._tenant = name, portable, tenant_id

    def export(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        if tenant_id != self._tenant:
            return {"instances": []}
        from ..privacy.privacy_memory import PrivacyMemory

        return {"instances": [m.export_subject(subject_id) for m in PrivacyMemory.live_instances()]}


class FileStoreRightsProvider:
    """Inventory (path, size, sha256) of files named by the subject tag; small text files are included verbatim,
    other small files base64; anything over ``max_inline_bytes`` is listed without content (never silently dropped)."""

    def __init__(self, root: str, *, name: str = "file_store", portable: bool = False, max_inline_bytes: int = 1_000_000) -> None:
        self.name, self.portable, self._root, self._max = name, portable, Path(root), int(max_inline_bytes)

    def export(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        tag = subject_tag(tenant_id, subject_id)
        files: List[Dict[str, Any]] = []
        if self._root.exists():
            for path in sorted(self._root.rglob("*")):
                inside = tag in path.relative_to(self._root).parts or path.name == tag or path.name.startswith(tag + ".")
                if not inside or not path.is_file() or path.is_symlink():
                    continue
                raw = path.read_bytes()
                row: Dict[str, Any] = {"path": path.relative_to(self._root).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                if len(raw) > self._max:
                    row["content"] = "omitted: larger than inline limit; provide through a secure channel"
                else:
                    try:
                        row["content_utf8"] = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        row["content_base64"] = base64.b64encode(raw).decode("ascii")
                files.append(row)
        return {"files": files}
