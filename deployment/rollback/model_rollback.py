import os
import shutil
import logging
import filecmp
import hashlib

from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Checksum validation (Rabin 1981 - Fingerprinting)
def generate_checksum(path: Path) -> str:
    """Create content-based hash for validation"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Backup verification (Petersen et al. 2008 - Data Integrity)
def validate_backup_integrity(src: Path, dst: Path) -> bool:
    """Three-layer validation for backup reliability"""
    if not filecmp.cmp(src, dst, shallow=False):
        return False
    if src.stat().st_size != dst.stat().st_size:
        return False
    return generate_checksum(src) == generate_checksum(dst)

def rollback_model(models_dir='models/', backup_dir='models/backups/'):
    """Implements reliable restore (Gray & Reuter 1993 - Transaction Processing)"""
    # Version existence check
    models_path = Path(models_dir)
    backup_path = Path(backup_dir)
    
    # 1. Backup current state
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_backup = backup_path / f"rollback_{timestamp}"
    current_backup.mkdir(parents=True, exist_ok=True)
    
    for item in models_path.iterdir():
        if item.name == 'backups':
            continue
        dest = current_backup / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # 2. Find latest pre-existing backup
    backups = sorted([d for d in backup_path.iterdir() if d.is_dir() and d.name.startswith("rollback_")], 
                    key=lambda x: x.name, reverse=True)
    if not backups:
        raise ValueError("No valid backups available")
    latest_backup = backups[0]
    
    # 3. Restore from backup
    for item in models_path.iterdir():
        if item.name == 'backups':
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    
    for item in latest_backup.iterdir():
        dest = models_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    logger.info(f"Rollback to {latest_backup.name} completed. Current state backed up to {current_backup.name}")
    return True

def rollforward_model(models_dir='models/', backup_dir='models/backups/', target_backup=None):
    models_path = Path(models_dir)
    backup_path = Path(backup_dir)

    if not backup_path.exists():
        raise FileNotFoundError(f"Backup directory {backup_dir} does not exist")

    if not any(backup_path.iterdir()):
        raise ValueError("Backup directory is empty")

    # 2. Find target backup
    backups = sorted([d for d in backup_path.iterdir() if d.is_dir() and d.name.startswith("rollback_")], 
                    key=lambda x: x.name, reverse=True)
    if not backups:
        raise ValueError("No valid backups available")
    
    if target_backup:
        target = next((b for b in backups if b.name == target_backup), None)
        if not target:
            raise ValueError(f"Backup {target_backup} not found")
    else:
        target = backups[0]

    # Find the most recent backup after current state
    current_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    newer_backups = [b for b in backups if b.name > f"rollback_{current_version}"]
    if not newer_backups:
        raise ValueError("No newer version to roll forward to")

    target_backup = newer_backups[-1]

    # Backup current state
    current_backup = backup_path / f"rollforward_{current_version}"
    current_backup.mkdir(parents=True, exist_ok=True)
    for item in models_path.iterdir():
        if item.name == 'backups':
            continue
        dest = current_backup / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Restore target
    for item in models_path.iterdir():
        if item.name == 'backups':
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for item in target_backup.iterdir():
        dest = models_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    logger.info(f"Rollforward to {target_backup.name} completed.")
    return True
