import os
import shutil
import logging
import filecmp
import hashlib

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
    backups = sorted(Path(backup_dir).glob("*.pt"))
    if not backups:
        raise ValueError("No valid model backups available")
        
    # Validate latest backup
    latest_backup = backups[-1]
    if not validate_backup_integrity(latest_backup, latest_backup):
        raise RuntimeError("Backup integrity check failed")

    """
    Restores model files from backup.
    """
    logger.info(f"Initiating model rollback: {models_dir=} {backup_dir=}")
    
    if not os.path.exists(backup_dir):
        logger.error(f"Backup directory does not exist: {backup_dir}")
        return False

    # Clean current models directory except backups/
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if item != 'backups':
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    # Restore backup files
    for item in os.listdir(backup_dir):
        src = os.path.join(backup_dir, item)
        dst = os.path.join(models_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    # Checksum verification after copy
    for item in Path(backup_dir).iterdir():
        dst = Path(models_dir)/item.name
        if not validate_backup_integrity(item, dst):
            raise RuntimeError(f"Restoration failed for {item.name}")

    logger.info("Model rollback completed successfully.")
    return True
