"""
Production-ready checkpoint management for SLAI training/inference workflows.

The manager owns orchestration. Reusable helper functions, error classes,
manifest serialization, atomic persistence, integrity checks, tokenizer logic,
and RNG utilities live in ``checkpoint_utils.py``.

Key properties
--------------
- Atomic checkpoint commits through staging directories.
- Manifest-based file integrity verification with SHA-256 hashes.
- PyTorch and NPZ checkpoint formats.
- Backward-compatible loading for legacy ``model_weights.pt`` and
  ``model_weights.npz`` layouts.
- Optional optimizer, scheduler, AMP scaler, RNG, metrics, and metadata support.
- Tokenizer persistence for simple custom tokenizers and common save/load APIs.
- Safe version handling to prevent path traversal and accidental overwrite.
- Optional retention cleanup and archive creation.

Public API examples
-------------------
    manager = CheckpointManager("src/checkpoints", create_archive=True)
    record = manager.save(model, tokenizer, optimizer=optim, epoch=3)
    state = manager.load(model, tokenizer, optimizer=optim, version="latest")
"""

from __future__ import annotations

import os
import shutil
import tempfile
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .checkpoint_utils import *
from .checkpoint_utils import read_manifest as read_manifest_file
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Checkpoint Manager")
printer = PrettyPrinter()


class CheckpointManager:
    """
    Save, load, list, verify, archive, and delete model checkpoints.

    Parameters
    ----------
    base_dir:
        Directory where checkpoint version directories are stored.
    default_format:
        Format used when ``save``/``load`` do not receive a format.
    allow_overwrite:
        Whether a save may replace an existing checkpoint version.
    create_archive:
        Whether saves should create a ``.tar.gz`` archive after committing.
    retention_limit:
        Optional maximum number of checkpoint directories to keep.
    """

    def __init__(
        self,
        base_dir: str | os.PathLike[str] = "src/checkpoints",
        *,
        default_format: CheckpointFormat = "torch",
        allow_overwrite: bool = False,
        create_archive: bool = False,
        retention_limit: Optional[int] = None,
    ) -> None:
        self.base_dir = ensure_directory(Path(base_dir))
        self.default_format = normalize_format(default_format)
        self.allow_overwrite = allow_overwrite
        self.create_archive = create_archive
        self.retention_limit = retention_limit
        logger.info("CheckpointManager initialized at %s", self.base_dir)

    # ------------------------------------------------------------------
    # Save API
    # ------------------------------------------------------------------
    def save(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        version: Optional[str] = None,
        format: Optional[str] = None,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        scaler: Any = None,
        epoch: Optional[int] = None,
        current_epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra_state: Optional[Mapping[str, Any]] = None,
        archive: Optional[bool] = None,
        overwrite: Optional[bool] = None,
        save_rng: bool = True,
        save_on_cpu: bool = True,
    ) -> CheckpointRecord:
        """
        Save a checkpoint and return its manifest record.

        ``current_epoch`` is accepted for compatibility with earlier code.
        New code should prefer ``epoch``.
        """
        checkpoint_format = normalize_format(format or self.default_format)
        epoch_value = epoch if epoch is not None else current_epoch

        if checkpoint_format == "torch":
            return self.save_torch(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch_value,
                step=step,
                metadata=metadata,
                metrics=metrics,
                extra_state=extra_state,
                version=version,
                archive=archive,
                overwrite=overwrite,
                save_rng=save_rng,
                save_on_cpu=save_on_cpu,
            )

        return self.save_npz(
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            metrics=metrics,
            version=version,
            epoch=epoch_value,
            step=step,
            archive=archive,
            overwrite=overwrite,
        )

    def save_torch(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        current_epoch: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        version: Optional[str] = None,
        *,
        scheduler: Any = None,
        scaler: Any = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra_state: Optional[Mapping[str, Any]] = None,
        archive: Optional[bool] = None,
        overwrite: Optional[bool] = None,
        save_rng: bool = True,
        save_on_cpu: bool = True,
    ) -> CheckpointRecord:
        """Save a full PyTorch training checkpoint."""
        version_tag = make_version(version)
        allow_overwrite = self.allow_overwrite if overwrite is None else overwrite
        final_dir, staging_dir = prepare_staging_dir(
            self.base_dir,
            version_tag,
            allow_overwrite=allow_overwrite,
        )
        epoch_value = epoch if epoch is not None else current_epoch

        try:
            payload: Dict[str, Any] = {
                "schema_version": 2,
                "format": "torch",
                "version": version_tag,
                "created_at": utc_now_iso(),
                "model_state": recursive_to_cpu(model.state_dict()) if save_on_cpu else model.state_dict(),
                "epoch": epoch_value,
                "step": step,
                "metadata": dict(metadata or {}),
                "metrics": dict(metrics or {}),
                "extra_state": dict(extra_state or {}),
            }

            if optimizer is not None:
                payload["optimizer_state"] = (
                    recursive_to_cpu(optimizer.state_dict()) if save_on_cpu else optimizer.state_dict()
                )
            if scheduler is not None and hasattr(scheduler, "state_dict"):
                payload["scheduler_state"] = scheduler.state_dict()
            if scaler is not None and hasattr(scaler, "state_dict"):
                payload["scaler_state"] = scaler.state_dict()
            if save_rng:
                payload["rng_state"] = capture_rng_state()

            atomic_torch_save(payload, staging_dir / TORCH_CHECKPOINT_NAME)
            tokenizer_kind = save_tokenizer(tokenizer, staging_dir)
            atomic_json_dump(dict(metadata or {}), staging_dir / METADATA_NAME)

            write_manifest(
                staging_dir,
                version=version_tag,
                checkpoint_format="torch",
                checkpoint_path=final_dir,
                epoch=epoch_value,
                step=step,
                metadata=dict(metadata or {}),
                metrics=dict(metrics or {}),
                tokenizer_kind=tokenizer_kind,
            )
            commit_staging_dir(staging_dir, final_dir, allow_overwrite=allow_overwrite)
            record = self.read_manifest(version_tag)
            assert record is not None, f"Manifest for version '{version_tag}' should exist after commit"
            self._post_save(version_tag, archive=archive)
            logger.info("Saved PyTorch checkpoint '%s' to %s", version_tag, final_dir)
            return record
        except Exception as exc:
            safe_rmtree(staging_dir)
            logger.exception("Failed to save PyTorch checkpoint '%s'", version_tag)
            if isinstance(exc, CheckpointError):
                raise
            raise CheckpointSaveError(f"Failed to save PyTorch checkpoint '{version_tag}': {exc}") from exc

    def save_npz(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        version: Optional[str] = None,
        *,
        metrics: Optional[Mapping[str, Any]] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        archive: Optional[bool] = None,
        overwrite: Optional[bool] = None,
        compressed: bool = True,
    ) -> CheckpointRecord:
        """
        Save model tensors in NPZ format.

        NPZ stores model tensors and tokenizer/metadata files. It does not
        store optimizer, scheduler, scaler, or RNG state; use torch format for
        resumable training checkpoints.
        """
        version_tag = make_version(version)
        allow_overwrite = self.allow_overwrite if overwrite is None else overwrite
        final_dir, staging_dir = prepare_staging_dir(
            self.base_dir,
            version_tag,
            allow_overwrite=allow_overwrite,
        )

        try:
            arrays = model_state_to_numpy(model)
            if compressed:
                np.savez_compressed(staging_dir / NPZ_WEIGHTS_NAME, **arrays)  # type: ignore[arg-type]
            else:
                np.savez(staging_dir / NPZ_WEIGHTS_NAME, **arrays)  # type: ignore[arg-type]

            tokenizer_kind = save_tokenizer(tokenizer, staging_dir)
            atomic_json_dump(dict(metadata or {}), staging_dir / METADATA_NAME)
            write_manifest(
                staging_dir,
                version=version_tag,
                checkpoint_format="npz",
                checkpoint_path=final_dir,
                epoch=epoch,
                step=step,
                metadata=dict(metadata or {}),
                metrics=dict(metrics or {}),
                tokenizer_kind=tokenizer_kind,
            )
            commit_staging_dir(staging_dir, final_dir, allow_overwrite=allow_overwrite)
            record = self.read_manifest(version_tag)
            assert record is not None, f"Manifest for version '{version_tag}' should exist after commit"
            self._post_save(version_tag, archive=archive)
            logger.info("Saved NPZ checkpoint '%s' to %s", version_tag, final_dir)
            return record
        except Exception as exc:
            safe_rmtree(staging_dir)
            logger.exception("Failed to save NPZ checkpoint '%s'", version_tag)
            if isinstance(exc, CheckpointError):
                raise
            raise CheckpointSaveError(f"Failed to save NPZ checkpoint '{version_tag}': {exc}") from exc

    # ------------------------------------------------------------------
    # Load API
    # ------------------------------------------------------------------
    def load(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        version: Optional[str] = None,
        format: Optional[str] = None,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: str | torch.device | Mapping[str, str] | None = "cpu",
        strict: bool = True,
        restore_rng: bool = False,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_scaler: bool = True,
        verify_integrity: bool = True,
    ) -> Dict[str, Any]:
        """Load a checkpoint into provided model/tokenizer/training objects."""
        resolved_version = self._resolve_version(version)
        record = self.read_manifest(resolved_version, missing_ok=True)
        checkpoint_format = normalize_format(format or (record.format if record else self.default_format))

        if checkpoint_format == "torch":
            return self.load_torch(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                version=resolved_version,
                map_location=map_location,
                strict=strict,
                restore_rng=restore_rng,
                load_optimizer=load_optimizer,
                load_scheduler=load_scheduler,
                load_scaler=load_scaler,
                verify_integrity=verify_integrity,
            )

        return self.load_npz(
            model=model,
            tokenizer=tokenizer,
            version=resolved_version,
            strict=strict,
            verify_integrity=verify_integrity,
        )

    def load_torch(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,  # kept for compatibility; ignored.
        version: Optional[str] = None,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: str | torch.device | Mapping[str, str] | None = "cpu",
        strict: bool = True,
        restore_rng: bool = False,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_scaler: bool = True,
        verify_integrity: bool = True,
    ) -> Dict[str, Any]:
        """Load a PyTorch checkpoint, including optional training state."""
        del metadata
        resolved_version = self._resolve_version(version)
        checkpoint_dir = self._checkpoint_dir(resolved_version)

        if verify_integrity:
            self.verify_checkpoint(resolved_version)

        checkpoint_file = first_existing(
            checkpoint_dir / TORCH_CHECKPOINT_NAME,
            checkpoint_dir / LEGACY_TORCH_CHECKPOINT_NAME,
        )
        if checkpoint_file is None:
            raise CheckpointLoadError(f"No PyTorch checkpoint file found in {checkpoint_dir}")

        payload = torch_load(checkpoint_file, map_location=map_location)
        if isinstance(payload, MutableMappingCompat) and "model_state" in payload:
            model_state = payload["model_state"]
        elif isinstance(payload, dict):
            model_state = payload
            payload = {
                "schema_version": 1,
                "format": "torch",
                "model_state": model_state,
                "metadata": {},
                "metrics": {},
            }
        else:
            raise CheckpointLoadError(f"Unsupported PyTorch checkpoint payload at {checkpoint_file}")

        incompatible = model.load_state_dict(model_state, strict=strict)

        if optimizer is not None and load_optimizer and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and load_scheduler and payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        if scaler is not None and load_scaler and payload.get("scaler_state") is not None:
            scaler.load_state_dict(payload["scaler_state"])
        if restore_rng and payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])

        loaded_tokenizer = load_tokenizer(tokenizer, checkpoint_dir)
        record = self.read_manifest(resolved_version, missing_ok=True)
        result = {
            "version": resolved_version,
            "path": str(checkpoint_dir),
            "format": "torch",
            "manifest": record.to_json_dict() if record else None,
            "epoch": payload.get("epoch"),
            "step": payload.get("step"),
            "metadata": payload.get("metadata") or {},
            "metrics": payload.get("metrics") or {},
            "extra_state": payload.get("extra_state") or {},
            "missing_keys": list(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
            "tokenizer": loaded_tokenizer,
        }
        logger.info("Loaded PyTorch checkpoint '%s' from %s", resolved_version, checkpoint_dir)
        return result

    def load_npz(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        version: Optional[str] = None,
        *,
        strict: bool = True,
        verify_integrity: bool = True,
    ) -> Dict[str, Any]:
        """Load model tensors from an NPZ checkpoint."""
        resolved_version = self._resolve_version(version)
        checkpoint_dir = self._checkpoint_dir(resolved_version)

        if verify_integrity:
            self.verify_checkpoint(resolved_version)

        compatibility = load_npz_into_model(model, checkpoint_dir / NPZ_WEIGHTS_NAME, strict=strict)
        loaded_tokenizer = load_tokenizer(tokenizer, checkpoint_dir)
        record = self.read_manifest(resolved_version, missing_ok=True)
        result = {
            "version": resolved_version,
            "path": str(checkpoint_dir),
            "format": "npz",
            "manifest": record.to_json_dict() if record else None,
            "epoch": record.epoch if record else None,
            "step": record.step if record else None,
            "metadata": record.metadata if record else {},
            "metrics": record.metrics if record else {},
            "tokenizer": loaded_tokenizer,
            **compatibility,
        }
        logger.info("Loaded NPZ checkpoint '%s' from %s", resolved_version, checkpoint_dir)
        return result

    # ------------------------------------------------------------------
    # Discovery, verification, archive, and deletion
    # ------------------------------------------------------------------
    def list_checkpoints(self) -> list[str]:
        """Return checkpoint version names sorted by creation time and name."""
        return [record.version for record in self.list_checkpoint_records()]

    def list_checkpoint_records(self) -> list[CheckpointRecord]:
        """Return checkpoint records, using manifest data when available."""
        records: list[CheckpointRecord] = []
        if not self.base_dir.exists():
            return records

        for path in self.base_dir.iterdir():
            if not path.is_dir() or path.name.startswith(".tmp-") or path.name.startswith(".replace-"):
                continue
            try:
                record = read_manifest_file(path, missing_ok=True)
                if record is not None:
                    records.append(record)
                elif looks_like_legacy_checkpoint(path):
                    records.append(build_legacy_record(path))
            except Exception as exc:
                logger.warning("Skipping unreadable checkpoint at %s: %s", path, exc)
        return sorted(records, key=lambda item: (item.created_at, item.version))

    def get_latest_checkpoint(self) -> Optional[str]:
        """Return the newest checkpoint version, or None when no checkpoints exist."""
        records = self.list_checkpoint_records()
        return records[-1].version if records else None

    def read_manifest(self, version: str, *, missing_ok: bool = False) -> Optional[CheckpointRecord]:
        """Read a checkpoint manifest by version."""
        checkpoint_dir = self._checkpoint_dir(version)
        return read_manifest_file(checkpoint_dir, missing_ok=missing_ok)

    def verify_checkpoint(self, version: str) -> bool:
        """Verify manifest-tracked files for a checkpoint."""
        safe_version = self._sanitize_or_resolve(version)
        checkpoint_dir = self._checkpoint_dir(safe_version)
        record = read_manifest_file(checkpoint_dir, missing_ok=True)

        if record is None:
            if looks_like_legacy_checkpoint(checkpoint_dir):
                logger.warning("Checkpoint '%s' has no manifest; skipping integrity verification", safe_version)
                return True
            raise FileNotFoundError(f"Checkpoint '{safe_version}' has no manifest and is not a legacy checkpoint")

        return verify_files_against_record(checkpoint_dir, record)

    def archive_checkpoint(self, version: str, *, overwrite: bool = True) -> Path:
        """Create a ``.tar.gz`` archive for a checkpoint directory and hash it."""
        safe_version = self._sanitize_or_resolve(version)
        return archive_checkpoint_dir(self.base_dir, safe_version, overwrite=overwrite)

    def delete_checkpoint(self, version: str, *, delete_archive: bool = True, missing_ok: bool = True) -> bool:
        """Delete a checkpoint directory and optionally its archive/hash."""
        safe_version = self._sanitize_or_resolve(version)
        checkpoint_dir = self._checkpoint_dir(safe_version)
        removed = False

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            removed = True
            logger.info("Deleted checkpoint directory %s", checkpoint_dir)
        elif not missing_ok:
            raise FileNotFoundError(f"Checkpoint '{safe_version}' not found at {checkpoint_dir}")

        if delete_archive:
            for path in (self.base_dir / f"{safe_version}.tar.gz", self.base_dir / f"{safe_version}.tar.gz.sha256"):
                if path.exists():
                    path.unlink()
                    removed = True
        return removed

    def cleanup_old_checkpoints(self, keep: Optional[int] = None) -> list[str]:
        """Keep the newest checkpoints and delete older ones."""
        keep_count = self.retention_limit if keep is None else keep
        if keep_count is None or keep_count < 0:
            return []

        records = self.list_checkpoint_records()
        if len(records) <= keep_count:
            return []

        to_delete = records[: max(0, len(records) - keep_count)]
        deleted: list[str] = []
        for record in to_delete:
            if self.delete_checkpoint(record.version):
                deleted.append(record.version)
        if deleted:
            logger.info("Deleted old checkpoints due to retention policy: %s", deleted)
        return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _post_save(self, version: str, *, archive: Optional[bool]) -> None:
        should_archive = self.create_archive if archive is None else archive
        if should_archive:
            self.archive_checkpoint(version)
        self.cleanup_old_checkpoints()

    def _resolve_version(self, version: Optional[str]) -> str:
        if version is None or version == "latest":
            latest = self.get_latest_checkpoint()
            if latest is None:
                raise FileNotFoundError("No checkpoints available")
            return latest
        return sanitize_version(version)

    def _sanitize_or_resolve(self, version: str) -> str:
        if version == "latest":
            latest = self.get_latest_checkpoint()
            if latest is None:
                raise FileNotFoundError("No checkpoints available")
            return latest
        return sanitize_version(version)

    def _checkpoint_dir(self, version: str) -> Path:
        safe_version = self._sanitize_or_resolve(version)
        return resolve_checkpoint_path(self.base_dir, safe_version)


# ``typing.MutableMapping`` cannot be used directly with ``isinstance`` on all
# Python/PyTorch environments through postponed annotations. Keep the runtime
# check explicit and local to avoid importing the name throughout the manager.
MutableMappingCompat = dict



__all__ = [
    "CheckpointError",
    "CheckpointFileInfo",
    "CheckpointIntegrityError",
    "CheckpointLoadError",
    "CheckpointManager",
    "CheckpointRecord",
    "CheckpointSaveError",
    "CheckpointVersionError",
]


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Script-level smoke test
    # ---------------------------------------------------------------------------
    class _TinyTokenizer:
        def __init__(self) -> None:
            self.word_to_id = {"<pad>": 0, "hello": 1, "world": 2}
            self.id_to_word = {0: "<pad>", 1: "hello", 2: "world"}
            self.vocab_size = len(self.word_to_id)


    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)


    def _run_self_test() -> None:
        print("\n=== Running Checkpoint Manager ===\n")
        printer.status("TEST", "Checkpoint Manager initialized", "info")

        with tempfile.TemporaryDirectory(prefix="checkpoint-manager-test-") as tmpdir:
            base_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(
                base_dir=base_dir,
                default_format="torch",
                allow_overwrite=False,
                create_archive=True,
                retention_limit=None,
            )

            torch.manual_seed(7)
            model = _TinyModel()
            tokenizer = _TinyTokenizer()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

            x = torch.randn(3, 4)
            y = torch.tensor([0, 1, 0])
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            printer.status("TEST", "Completed tiny training step", "success")

            torch_record = manager.save(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                step=10,
                metrics={"loss": float(loss.detach().cpu())},
                metadata={"purpose": "self_test", "format": "torch"},
                version="unit_torch",
                format="torch",
            )
            _assert(torch_record.version == "unit_torch", "Torch version mismatch")
            _assert((base_dir / "unit_torch" / MANIFEST_NAME).exists(), "Torch manifest missing")
            _assert((base_dir / "unit_torch.tar.gz").exists(), "Torch archive missing")
            manager.verify_checkpoint("unit_torch")
            printer.status("TEST", "Saved and verified torch checkpoint", "success")

            restored_model = _TinyModel()
            restored_tokenizer = _TinyTokenizer()
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.001)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
            loaded_torch = manager.load(
                model=restored_model,
                tokenizer=restored_tokenizer,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                version="unit_torch",
                format="torch",
                strict=True,
            )
            _assert(loaded_torch["epoch"] == 1, "Torch epoch was not restored")
            _assert(restored_tokenizer.vocab_size == tokenizer.vocab_size, "Tokenizer vocab size mismatch")
            _assert(not loaded_torch["missing_keys"], "Unexpected missing keys in torch load")
            _assert(not loaded_torch["unexpected_keys"], "Unexpected keys in torch load")
            printer.status("TEST", "Loaded torch checkpoint", "success")

            npz_record = manager.save(
                model=model,
                tokenizer=tokenizer,
                metadata={"purpose": "self_test", "format": "npz"},
                metrics={"loss": float(loss.detach().cpu())},
                version="unit_npz",
                format="npz",
                archive=False,
            )
            _assert(npz_record.version == "unit_npz", "NPZ version mismatch")
            manager.verify_checkpoint("unit_npz")
            printer.status("TEST", "Saved and verified NPZ checkpoint", "success")

            npz_model = _TinyModel()
            npz_tokenizer = _TinyTokenizer()
            loaded_npz = manager.load_npz(
                model=npz_model,
                tokenizer=npz_tokenizer,
                version="unit_npz",
                strict=True,
            )
            _assert(loaded_npz["version"] == "unit_npz", "NPZ load version mismatch")
            _assert(not loaded_npz["mismatched_keys"], "Unexpected mismatched keys in NPZ load")
            printer.status("TEST", "Loaded NPZ checkpoint", "success")

            versions = manager.list_checkpoints()
            _assert("unit_torch" in versions and "unit_npz" in versions, "Checkpoint listing incomplete")
            latest = manager.get_latest_checkpoint()
            _assert(latest in {"unit_torch", "unit_npz"}, "Latest checkpoint resolution failed")
            printer.pretty("CHECKPOINTS", [checkpoint_summary(r) for r in manager.list_checkpoint_records()], "info")

            try:
                manager.read_manifest("../escape")
                raise AssertionError("Unsafe checkpoint version was not rejected")
            except CheckpointVersionError:
                printer.status("TEST", "Rejected unsafe checkpoint version", "success")

            _assert(manager.delete_checkpoint("unit_npz"), "NPZ checkpoint was not deleted")
            _assert("unit_npz" not in manager.list_checkpoints(), "Deleted checkpoint still listed")
            printer.status("TEST", "Deleted checkpoint successfully", "success")

    print("\n=== Test ran successfully ===\n")


    _run_self_test()
