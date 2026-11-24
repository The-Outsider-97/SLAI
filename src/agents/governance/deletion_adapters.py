"""Deletion adapters binding the PrivacyLedger to SLAI's real data stores (audit 4.4).

Each adapter performs a real deletion on its store AND independently verifies absence, because
the ledger only reports an erasure request as ``completed`` when every registered adapter's
``verify_absent`` is True.

Linkage convention (how a stored item is attributed to a data subject):
  * Producers tag SharedMemory / SecureMemory items with ``subject_tag(tenant_id, subject_id)``.
    The tag is a salt-free hash, so it contains no personal data and survives identifier
    normalisation. Items that are not tagged cannot be attributed and therefore cannot be erased
    by subject: producers of personal data MUST tag it (this is a documented integration duty).
  * PrivacyMemory already records ``subject_id`` and is erased through its own index.

Known limits (reported, not hidden): copies held by external recipients, model weights, logs
outside the sanitised pipeline and offline backups are not reachable from here; backups are
covered only by the ledger's restore-time tombstone check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_TAG_PREFIX = "subj_"


def subject_tag(tenant_id: str, subject_id: str) -> str:
    """Deterministic, PII-free tag attributing a stored item to (tenant, subject)."""
    if not tenant_id or not subject_id:
        raise ValueError("tenant_id and subject_id are required")
    return _TAG_PREFIX + hashlib.sha256(f"{tenant_id}\x1f{subject_id}".encode("utf-8")).hexdigest()[:40]


class SharedMemoryDeletionAdapter:
    """Erases tagged SharedMemory items (all versions) and refreshes the persisted state file."""

    def __init__(self, memory: Any, *, name: str = "shared_memory", persist_path: Optional[str] = None) -> None:
        self.name = name
        self._m = memory
        self._persist = Path(persist_path) if persist_path else None

    def _matches(self, tag: str) -> List[Mapping[str, Any]]:
        return list(self._m.get_by_tag(tag) or [])

    def delete(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        tag = subject_tag(tenant_id, subject_id)
        removed = 0
        for item in self._matches(tag):
            if self._m.delete(item["key"]):
                removed += 1
        persisted = False
        if self._persist is not None:
            # Overwrite the sealed state file so the erased items do not survive on disk.
            self._m.save_to_file(str(self._persist))
            persisted = True
        return {"removed_keys": removed, "persisted_state_rewritten": persisted}

    def verify_absent(self, tenant_id: str, subject_id: str) -> bool:
        tag = subject_tag(tenant_id, subject_id)
        if self._matches(tag):
            return False
        if self._persist is not None:
            return self._persisted_clean(tag)
        return True

    def _persisted_clean(self, tag: str) -> bool:
        if not self._persist.exists():
            return True
        from ..collaborative import shared_memory as sm  # lazy: heavy module
        from ..collaborative.utils import secure_persistence as sp

        payload = sp.unseal(self._persist.read_bytes(), versioned_item=sm.VersionedItem)  # raises if tampered/unreadable => caller treats as failed
        return not any(tag in (tags or []) for tags in dict(payload.get("tags", {})).values())


class SecureMemoryDeletionAdapter:
    """Erases tagged SecureMemory entries for one tenant and purges on-disk checkpoints that hold them."""

    def __init__(self, memory: Any, *, name: str = "secure_memory", purge_checkpoints: bool = True) -> None:
        self.name = name
        self._m = memory
        self._purge = purge_checkpoints

    def _ctx(self, tenant_id: str) -> Dict[str, Any]:
        return self._m.internal_context("subject_erasure", principal="governance_deletion", tenant_id=tenant_id)

    def _entry_ids(self, tag: str) -> List[str]:
        return list(self._m.tag_index.get(tag, set()))

    def delete(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        tag = subject_tag(tenant_id, subject_id)
        ctx = self._ctx(tenant_id)
        removed, held = 0, 0
        for entry_id in self._entry_ids(tag):
            entry = self._m.store.get(entry_id) or {}
            if (entry.get("meta") or {}).get("legal_hold"):
                held += 1  # entry-level hold: never deleted here; verification will fail => request stays incomplete
                continue
            if self._m.delete(entry_id, ctx, reason="data_subject_erasure"):
                removed += 1
        purged = self._purge_checkpoints(tag) if self._purge else 0
        if purged:
            self._m.create_checkpoint()  # fresh checkpoint without the erased entries
        return {"removed_entries": removed, "skipped_legal_hold": held, "checkpoints_purged": purged}

    def _checkpoint_hits(self, tag: str) -> List[Path]:
        hits = []
        for path in self._m.checkpoint_files():
            try:
                if tag in path.read_text(encoding="utf-8", errors="ignore"):
                    hits.append(path)
            except OSError:
                hits.append(path)  # unreadable => cannot prove absence
        return hits

    def _purge_checkpoints(self, tag: str) -> int:
        purged = 0
        for path in self._checkpoint_hits(tag):
            path.unlink()
            purged += 1
        return purged

    def verify_absent(self, tenant_id: str, subject_id: str) -> bool:
        tag = subject_tag(tenant_id, subject_id)
        if self._entry_ids(tag):
            return False
        return not self._checkpoint_hits(tag)


class PrivacyMemoryDeletionAdapter:
    """Erases a tenant-bound PrivacyMemory (consent, bindings, decisions, retention, audit events).

    ``PrivacyMemory`` itself is not tenant-aware, so the adapter is bound to ONE tenant. Requests for
    other tenants are no-ops (nothing of theirs lives in this instance) - do not bind one instance to
    several tenants. Retention records under an active legal hold are kept and make verification fail.

    Every privacy component (consent, id, minimisation, retention, auditability) owns its OWN PrivacyMemory.
    With ``memory=None`` the adapter therefore covers *all live instances* (``PrivacyMemory.live_instances()``),
    evaluated at call time so instances created later are covered too.
    """

    def __init__(self, memory: Any = None, *, tenant_id: str, name: str = "privacy_memory", held_record_ids=lambda tenant, subject: ()) -> None:
        self.name = name
        self._m = memory
        self._tenant = tenant_id
        self._held = held_record_ids

    def _memories(self) -> List[Any]:
        if self._m is not None:
            return [self._m]
        from ..privacy.privacy_memory import PrivacyMemory  # lazy: heavy module

        return PrivacyMemory.live_instances()

    def delete(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        if tenant_id != self._tenant:
            return {"skipped": "different_tenant"}
        skip = tuple(self._held(tenant_id, subject_id))
        memories = self._memories()
        totals: Dict[str, int] = {}
        for m in memories:
            for k, v in m.erase_subject(subject_id, skip_record_ids=skip).items():
                totals[k] = totals.get(k, 0) + int(v)
        totals["instances"] = len(memories)
        return totals

    def verify_absent(self, tenant_id: str, subject_id: str) -> bool:
        if tenant_id != self._tenant:
            return True
        skip = tuple(self._held(tenant_id, subject_id))
        return not any(m.contains_subject(subject_id, ignore_record_ids=skip) for m in self._memories())


class FileStoreDeletionAdapter:
    """Erases subject data kept in files: any entry under ``root`` whose name starts with the subject tag.

    Convention: a file or directory holding a subject's data is named ``<subject_tag>`` or ``<subject_tag>.<suffix>``
    (``subject_tag`` is PII-free). Deletion unlinks the files; it cannot guarantee physical erasure on
    SSD/journalling/copy-on-write filesystems or in snapshots - that is a deployment control (full-disk encryption
    plus key destruction) and is listed as such in the deployment documentation.
    """

    def __init__(self, root: str, *, name: str = "file_store") -> None:
        self.name = name
        self._root = Path(root)

    def _hits(self, tag: str) -> List[Path]:
        if not self._root.exists():
            return []
        return [p for p in self._root.rglob("*") if p.name == tag or p.name.startswith(tag + ".")]

    def delete(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        import shutil

        tag = subject_tag(tenant_id, subject_id)
        removed = 0
        for path in sorted(self._hits(tag), key=lambda p: len(p.parts)):
            if not path.exists():
                continue  # already removed with a parent directory
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        return {"removed_paths": removed}

    def verify_absent(self, tenant_id: str, subject_id: str) -> bool:
        return not self._hits(subject_tag(tenant_id, subject_id))
