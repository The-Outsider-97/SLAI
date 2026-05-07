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

import concurrent.futures
import os
import shutil
import tempfile
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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

    def __init__(self, base_dir: str | os.PathLike[str] = "src/checkpoints", *,
        default_format: CheckpointFormat = "torch",
        allow_overwrite: bool = False, create_archive: bool = False,
        retention_limit: Optional[int] = None,
    ) -> None:
        self.base_dir = ensure_directory(Path(base_dir))
        self.default_format = normalize_format(default_format)
        self.allow_overwrite = allow_overwrite
        self.create_archive = create_archive
        self.retention_limit = retention_limit
        logger.info("CheckpointManager initialized at %s", self.base_dir)

    def save_async(self, model: torch.nn.Module, tokenizer: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        version: Optional[str] = None, format: Optional[str] = None, *,
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
        executor: Optional[concurrent.futures.Executor] = None,
    ) -> concurrent.futures.Future:
        """
        Asynchronous version of `save()`.

        Returns a Future that resolves to the CheckpointRecord (or raises an
        exception on failure). If no executor is provided, a default thread
        pool is created and reused.
        """
        if executor is None:
            if not hasattr(self, "_default_executor"):
                self._default_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor = self._default_executor

        future = executor.submit(
            self.save,
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            version=version,
            format=format,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            current_epoch=current_epoch,
            step=step,
            metrics=metrics,
            extra_state=extra_state,
            archive=archive,
            overwrite=overwrite,
            save_rng=save_rng,
            save_on_cpu=save_on_cpu,
        )
        logger.debug("Submitted asynchronous save for version %s", version or "auto")
        return future

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
        load_components: Optional[Sequence[str]] = None,
        skip_components: Optional[Sequence[str]] = None,
        load_key_prefix: Optional[str] = None,
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
                load_components=load_components,
                skip_components=skip_components,
                load_key_prefix=load_key_prefix,
            )

        return self.load_npz(
            model=model,
            tokenizer=tokenizer,
            version=resolved_version,
            strict=strict,
            verify_integrity=verify_integrity,
            load_key_prefix=load_key_prefix,
            skip_components=skip_components,
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
        auto_repair: bool = False,
        load_components: Optional[Sequence[str]] = None,
        skip_components: Optional[Sequence[str]] = None,
        load_key_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a PyTorch checkpoint, optionally restricting which components are loaded.
    
        `load_components` can be a list like ["model_state", "optimizer_state"].
        `skip_components` overrides `load_components` if both are given.
        `load_key_prefix` filters model state dict keys by prefix (useful for submodules).
        """
        del metadata
        resolved_version = self._resolve_version(version)
        checkpoint_dir = self._checkpoint_dir(resolved_version)
    
        if verify_integrity:
            try:
                self.verify_checkpoint(resolved_version)
            except CheckpointIntegrityError as e:
                if auto_repair:
                    logger.warning("Integrity check failed for '%s': %s. Attempting repair from archive.", resolved_version, e)
                    try:
                        self._restore_from_archive(resolved_version)
                    except Exception as repair_err:
                        raise CheckpointLoadError(f"Auto‑repair failed for '{resolved_version}': {repair_err}") from e
                else:
                    raise
    
        checkpoint_file = first_existing(
            checkpoint_dir / TORCH_CHECKPOINT_NAME,
            checkpoint_dir / LEGACY_TORCH_CHECKPOINT_NAME,
        )
        if checkpoint_file is None:
            raise CheckpointLoadError(f"No PyTorch checkpoint file found in {checkpoint_dir}")
    
        payload = torch_load(checkpoint_file, map_location=map_location)
        # Normalise legacy payloads (same as existing code) ...
    
        # Determine which components to load
        allowed = None
        if load_components is not None:
            allowed = set(load_components)
        if skip_components is not None:
            disallowed = set(skip_components)
            if allowed is None:
                # Default set of all possible components
                all_components = {"model_state", "optimizer_state", "scheduler_state", "scaler_state", "rng_state"}
                allowed = all_components - disallowed
            else:
                allowed -= disallowed
    
        incompatible: Any = None

        # Load model state with optional prefix filtering
        if allowed is None or "model_state" in allowed:
            model_state = payload.get("model_state")
            if model_state is not None:
                if load_key_prefix:
                    model_state = {key: value for key, value in model_state.items() if key.startswith(load_key_prefix)}
                incompatible = model.load_state_dict(model_state, strict=strict)
            else:
                incompatible = {"missing_keys": list(model.state_dict().keys()), "unexpected_keys": []}
    
        # Conditionally load other components
        if (allowed is None or "optimizer_state" in allowed) and optimizer is not None and payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if (allowed is None or "scheduler_state" in allowed) and scheduler is not None and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        if (allowed is None or "scaler_state" in allowed) and scaler is not None and payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        if (allowed is None or "rng_state" in allowed) and restore_rng and payload.get("rng_state"):
            restore_rng_state(payload["rng_state"])
    
        loaded_tokenizer = None
        if allowed is None or "tokenizer" in allowed:
            loaded_tokenizer = load_tokenizer(tokenizer, checkpoint_dir)
    
        # Build result dictionary (same as before)
        record = self.read_manifest(resolved_version, missing_ok=True)
        logger.info("Loaded PyTorch checkpoint '%s' from %s", resolved_version, checkpoint_dir)
        return {
            "version": resolved_version,
            "path": str(checkpoint_dir),
            "format": "torch",
            "manifest": record.to_json_dict() if record else None,
            "epoch": payload.get("epoch"),
            "step": payload.get("step"),
            "metadata": payload.get("metadata") or {},
            "metrics": payload.get("metrics") or {},
            "extra_state": payload.get("extra_state") or {},
            "missing_keys": list(getattr(incompatible, "missing_keys", [])) if incompatible is not None else [],
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])) if incompatible is not None else [],
            "tokenizer": loaded_tokenizer,
        }

    def load_npz(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        version: Optional[str] = None,
        *,
        strict: bool = True,
        verify_integrity: bool = True,
        load_key_prefix: Optional[str] = None,
        skip_components: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Load model tensors from an NPZ checkpoint."""
        resolved_version = self._resolve_version(version)
        checkpoint_dir = self._checkpoint_dir(resolved_version)
    
        if verify_integrity:
            self.verify_checkpoint(resolved_version)
    
        # Load model weights with optional key prefix filtering
        weight_path = checkpoint_dir / NPZ_WEIGHTS_NAME
        if not weight_path.exists():
            raise CheckpointLoadError(f"NPZ weights file not found: {weight_path}")
    
        # Use load_npz_into_model with prefix filtering
        # Since load_npz_into_model doesn't support prefix, we do it ourselves
        state = model.state_dict()
        loaded: set[str] = set()
        unexpected: list[str] = []
        mismatched: list[Dict[str, Any]] = []
    
        with np.load(weight_path, allow_pickle=False) as weights:
            for name in weights.files:
                if load_key_prefix and not name.startswith(load_key_prefix):
                    continue
                target_name = name  # keep the full key; target model uses the same names
                if target_name not in state:
                    unexpected.append(name)
                    continue
                source = torch.as_tensor(weights[name], dtype=state[target_name].dtype, device=state[target_name].device)
                if tuple(source.shape) != tuple(state[target_name].shape):
                    mismatched.append({
                        "name": name,
                        "checkpoint_shape": tuple(source.shape),
                        "model_shape": tuple(state[target_name].shape),
                    })
                    continue
                state[target_name] = source
                loaded.add(target_name)
    
        missing = [name for name in state.keys() if name not in loaded]
        if strict and (missing or unexpected or mismatched):
            raise CheckpointLoadError(
                "NPZ checkpoint is incompatible with the model. "
                f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
            )
        incompatible = model.load_state_dict(state, strict=False)
    
        # Load tokenizer only if not skipped
        loaded_tokenizer = None
        if skip_components is None or "tokenizer" not in skip_components:
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
            "missing_keys": missing or list(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": unexpected or list(getattr(incompatible, "unexpected_keys", [])),
            "mismatched_keys": mismatched,
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
            raise CheckpointManifestError(f"Checkpoint '{safe_version}' has no manifest and is not a legacy checkpoint", path=checkpoint_dir, version=safe_version)

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
    def _restore_from_archive(self, version: str) -> Path:
        """Restore a checkpoint directory from its .tar.gz archive. Returns the restored directory path."""
        safe_version = self._sanitize_or_resolve(version)
        archive_path = self.base_dir / f"{safe_version}.tar.gz"
        if not archive_path.exists():
            raise CheckpointIntegrityError(f"No archive found for checkpoint '{safe_version}' at {archive_path}")
    
        # Optionally verify archive hash before extraction
        hash_path = self.base_dir / f"{safe_version}.tar.gz.sha256"
        if hash_path.exists():
            expected = hash_path.read_text().strip()
            actual = sha256_file(archive_path)
            if expected != actual:
                raise CheckpointIntegrityError(f"Archive SHA256 mismatch for '{safe_version}'")
    
        # Remove existing (broken) checkpoint directory and extract
        checkpoint_dir = self._checkpoint_dir(version)
        safe_rmtree(checkpoint_dir)
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    
        import tarfile

        def _validate_member(member: tarfile.TarInfo) -> None:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise CheckpointIntegrityError(
                    f"Unsafe archive member path '{member.name}' for checkpoint '{safe_version}'"
                )
            if member.issym() or member.islnk():
                raise CheckpointIntegrityError(
                    f"Refusing to extract link member '{member.name}' for checkpoint '{safe_version}'"
                )

        tmp_extract = Path(tempfile.mkdtemp(dir=self.base_dir, prefix=f".restore-{safe_version}-"))
        restore_dir = tmp_extract / safe_version
        restore_dir.mkdir(parents=True, exist_ok=False)
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                members = tar.getmembers()
                for member in members:
                    _validate_member(member)
                tar.extractall(path=restore_dir, members=members)

            extracted_items = list(restore_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                extracted_items[0].rename(checkpoint_dir)
            else:
                restore_dir.rename(checkpoint_dir)
        finally:
            safe_rmtree(tmp_extract)
    
        logger.info("Restored checkpoint '%s' from archive %s", safe_version, archive_path)
        return checkpoint_dir

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
    print("\n=== Running Checkpoint Manager Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting enhanced checkpoint tests", "info")
    from src.agents.perception.modules.tokenizer import Tokenizer # pyright: ignore[reportMissingImports]

    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 2),
            )
            self.extra = torch.nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.extra(self.net(x))

    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def _corrupt_checkpoint(version_dir: Path, file_to_corrupt: str = "checkpoint.pt") -> None:
        """Deliberately corrupt a file inside a checkpoint directory."""
        target = version_dir / file_to_corrupt
        if target.exists():
            with open(target, "ab") as f:
                f.write(b"CORRUPTED_DATA")
            print(f"  Corrupted {target}")


    with tempfile.TemporaryDirectory(prefix="checkpoint-manager-test-") as tmpdir:
        base_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(
            base_dir=base_dir,
            default_format="torch",
            allow_overwrite=False,
            create_archive=True,          # archives are needed for auto-repair
            retention_limit=None,
        )

        torch.manual_seed(7)
        model = _TinyModel()
        tokenizer = Tokenizer()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # One forward/backward step to make state non‑trivial
        x = torch.randn(3, 4)
        y = torch.tensor([0, 1, 0])
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        printer.status("TEST", "Prepared model, optimizer, scheduler", "success")

        # ------------------------------------------------------------------
        # 1. Asynchronous save
        # ------------------------------------------------------------------
        printer.status("TEST", "Testing async save (torch format)", "info")
        future = manager.save_async(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            step=20,
            metrics={"loss": float(loss.detach().cpu())},
            metadata={"test": "async"},
            version="async_torch",
            format="torch",
        )
        record_async = future.result(timeout=30)
        _assert(record_async.version == "async_torch", "Async save version mismatch")
        _assert((base_dir / "async_torch" / MANIFEST_NAME).exists(), "Async manifest missing")
        _assert((base_dir / "async_torch.tar.gz").exists(), "Async archive missing")
        printer.status("TEST", "Async save completed successfully", "success")

        # ------------------------------------------------------------------
        # 2. Selective loading (load only model and tokenizer, skip optimizer)
        # ------------------------------------------------------------------
        printer.status("TEST", "Testing selective loading (skip optimizer)", "info")
        model2 = _TinyModel()
        tokenizer2 = Tokenizer()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.999)  # different LR
        load_result = manager.load(
            model=model2,
            tokenizer=tokenizer2,
            optimizer=optimizer2,
            version="async_torch",
            format="torch",
            skip_components=["optimizer_state"],
            verify_integrity=True,
        )
        # Optimizer state should NOT be restored -> LR remains 0.999
        _assert(optimizer2.param_groups[0]["lr"] == 0.999, "Optimizer was incorrectly restored")
        _assert(tokenizer2.vocab_size == tokenizer.vocab_size, "Tokenizer not restored")
        _assert("missing_keys" not in load_result or not load_result["missing_keys"], "Unexpected missing keys")
        printer.status("TEST", "Selective loading (skip_components) works", "success")

        # ------------------------------------------------------------------
        # 3. Prefix loading (load only "net." submodule)
        # ------------------------------------------------------------------
        printer.status("TEST", "Testing load_key_prefix (load only 'net.')", "info")
        model_prefix = _TinyModel()
        # First, manually zero out the 'extra' layer to verify it stays untouched
        with torch.no_grad():
            model_prefix.extra.weight.fill_(0.0)
            model_prefix.extra.bias.fill_(0.0)

        load_result_prefix = manager.load(
            model=model_prefix,
            version="async_torch",
            format="torch",
            load_key_prefix="net.",
            strict=False,
        )
        # Check that 'net' weights were loaded (not zero) while 'extra' remains zero
        net_changed = not torch.allclose(model_prefix.net[0].weight, torch.zeros_like(model_prefix.net[0].weight)) # pyright: ignore[reportArgumentType]
        extra_unchanged = torch.allclose(model_prefix.extra.weight, torch.zeros_like(model_prefix.extra.weight))
        _assert(net_changed, "Prefix loading did not load 'net' weights")
        _assert(extra_unchanged, "Prefix loading incorrectly modified 'extra' layer")
        printer.status("TEST", "load_key_prefix works correctly", "success")

        # ------------------------------------------------------------------
        # 4. NPZ with prefix and skip_components
        # ------------------------------------------------------------------
        printer.status("TEST", "Testing NPZ save/load with prefix and skip_components", "info")
        npz_version = "test_npz_prefix"
        manager.save_npz(
            model=model,
            tokenizer=tokenizer,
            version=npz_version,
            metadata={"format": "npz"},
            compressed=True,
        )
        npz_model = _TinyModel()
        # Zero out extra layer again
        with torch.no_grad():
            npz_model.extra.weight.fill_(0.0)
        load_npz_prefix = manager.load_npz(
            model=npz_model,
            version=npz_version,
            load_key_prefix="net.",
            skip_components=["tokenizer"],
            strict=False,
        )
        net_loaded = not torch.allclose(npz_model.net[0].weight, torch.zeros_like(npz_model.net[0].weight)) # pyright: ignore[reportArgumentType]
        extra_still_zero = torch.allclose(npz_model.extra.weight, torch.zeros_like(npz_model.extra.weight))
        _assert(net_loaded, "NPZ prefix loading failed for 'net'")
        _assert(extra_still_zero, "NPZ prefix loading modified 'extra'")
        _assert(load_npz_prefix.get("tokenizer") is None, "Tokenizer was not skipped")
        printer.status("TEST", "NPZ with prefix and skip_components works", "success")

        # ------------------------------------------------------------------
        # 5. Auto‑repair from archive
        # ------------------------------------------------------------------
        printer.status("TEST", "Testing auto‑repair from archive", "info")
        # First create a healthy checkpoint (torch format)
        manager.save_torch(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            epoch=3,
            version="repair_me",
            archive=True,
        )
        # Corrupt the manifest (or any tracked file) inside the checkpoint directory
        ckpt_dir = base_dir / "repair_me"
        _corrupt_checkpoint(ckpt_dir, MANIFEST_NAME)

        # Attempt to load with auto_repair=True – should restore from archive
        repaired_model = _TinyModel()
        repaired_tokenizer = Tokenizer()
        try:
            load_repaired = manager.load_torch(
                model=repaired_model,
                tokenizer=repaired_tokenizer,
                version="repair_me",
                verify_integrity=True,
                auto_repair=True,
            )
            # After repair, the manifest must exist and be valid
            _assert((ckpt_dir / MANIFEST_NAME).exists(), "Auto‑repair did not restore manifest")
            _assert(manager.verify_checkpoint("repair_me"), "Checkpoint still corrupted after repair")
            printer.status("TEST", "Auto‑repair successfully restored checkpoint from archive", "success")
        except Exception as e:
            printer.status("TEST", f"Auto‑repair failed: {e}", "error")
            raise

        # ------------------------------------------------------------------
        # 6. Negative test: auto_repair=False raises integrity error
        # ------------------------------------------------------------------
        # Corrupt again
        _corrupt_checkpoint(ckpt_dir, MANIFEST_NAME)
        try:
            manager.load_torch(
                model=repaired_model,
                tokenizer=repaired_tokenizer,
                version="repair_me",
                verify_integrity=True,
                auto_repair=False,
            )
            raise AssertionError("Expected CheckpointIntegrityError but none was raised")
        except CheckpointIntegrityError:
            printer.status("TEST", "Integrity error correctly raised when auto_repair=False", "success")

        # ------------------------------------------------------------------
        # 7. Cleanup and retention test
        # ------------------------------------------------------------------
        printer.status("TEST", "Testing cleanup_old_checkpoints", "info")
        # Create a few more checkpoints
        for i in range(3):
            manager.save_torch(model, tokenizer, version=f"dummy_{i}", archive=False)
        # Set retention limit to 2 (oldest two should be deleted)
        manager.retention_limit = 2
        deleted = manager.cleanup_old_checkpoints()
        remaining = manager.list_checkpoints()
        _assert(len(remaining) == 2, f"Expected 2 checkpoints after cleanup, got {len(remaining)}")
        _assert("dummy_0" not in remaining, "Oldest checkpoint not deleted")
        printer.status("TEST", "Retention policy works", "success")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        printer.pretty("FINAL CHECKPOINTS", [
            checkpoint_summary(r) for r in manager.list_checkpoint_records()
        ], "info")
        printer.status("ALL TESTS", "CheckpointManager enhancements verified successfully", "success")

    print("\n=== All tests passed ===\n")
