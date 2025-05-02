"""
BASE_INFRA.PY - Core Infrastructure Components
Implements version control, rollback, and hyperparameter tuning foundations
"""

import subprocess
import hashlib
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import shared modules
from deployment.rollback.code_rollback import AtomicRollback, reset_to_commit, delete_tag
from deployment.rollback.model_rollback import rollback_model, validate_backup_integrity
from src.tuning.tuner import HyperparamTuner as BaseTuner
from src.tuning.bayesian_search import BayesianSearch
from logs.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

class RollbackSystem:
    """Implements atomic rollback operations across code and model versions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_dir = Path(config.get('rollback', {}).get('backup_dir', 'models/backups/'))
        self.model_dir = Path(config.get('paths', {}).get('models', 'models/'))
        self._verify_directories()

    def _verify_directories(self):
        """Ensure required directories exist with proper permissions"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def full_rollback(self, commit_hash: str, model_version: str) -> bool:
        """
        Execute atomic rollback of both code and model versions
        Implements 2-phase commit protocol for consistency
        """
        with AtomicRollback() as arb:
            # Phase 1: Prepare both rollbacks
            code_ok = self._rollback_code_prepare(commit_hash)
            model_ok = self._rollback_model_prepare(model_version)

            if not (code_ok and model_ok):
                logger.error("Rollback preparation failed")
                return False

            # Phase 2: Commit both rollbacks
            return self._commit_rollbacks(commit_hash, model_version)

    def _rollback_code_prepare(self, commit_hash: str) -> bool:
        """Validate and stage code rollback"""
        try:
            # Verify commit exists
            subprocess.run(['git', 'cat-file', '-e', commit_hash], check=True)
            logger.info(f"Validated commit {commit_hash} for rollback")
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Invalid commit hash: {commit_hash}")
            return False

    def _rollback_model_prepare(self, model_version: str) -> bool:
        """Validate model backup integrity"""
        model_path = self.backup_dir / f"{model_version}.pt"
        current_model = self.model_dir / "current.pt"
        
        if not validate_backup_integrity(model_path, current_model):
            logger.error(f"Model backup integrity check failed for {model_version}")
            return False
        return True

    def _commit_rollbacks(self, commit_hash: str, model_version: str) -> bool:
        """Execute the actual rollback operations"""
        try:
            # Reset code
            reset_to_commit(commit_hash, hard=True)
            # Restore model
            rollback_model(str(self.model_dir), str(self.backup_dir))
            logger.info(f"Successfully rolled back to commit {commit_hash} and model {model_version}")
            return True
        except Exception as e:
            logger.error(f"Rollback commit failed: {str(e)}")
            return False

class HyperparamTuner(BaseTuner):
    """Extended hyperparameter tuner with integrated rollback and safety checks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollback = RollbackSystem(kwargs.get('config', {}))
        self.best_state = None

    def _create_snapshot(self) -> Dict[str, Any]:
        """Capture current system state for potential rollback"""
        return {
            'model_hash': self._model_checksum(),
            'commit_hash': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
            'timestamp': datetime.now().isoformat()
        }

    def _model_checksum(self) -> str:
        """Generate content hash for current model"""
        model_file = self.rollback.model_dir / "current.pt"
        return hashlib.sha256(model_file.read_bytes()).hexdigest()

    def run_safe_tuning(self):
        """Execute tuning with automatic rollback on failure"""
        initial_state = self._create_snapshot()
        
        try:
            result = super().run_tuning_pipeline()
            if result['score'] < self.config.get('tuning', {}).get('min_score', 0.7):
                raise ValueError("Tuning resulted in suboptimal model")
            return result
        except Exception as e:
            logger.error(f"Tuning failed, initiating rollback: {str(e)}")
            self._execute_rollback(initial_state)
            raise

    def _execute_rollback(self, snapshot: Dict[str, Any]):
        """Restore system state from snapshot"""
        logger.info("Initiating tuning rollback...")
        success = self.rollback.full_rollback(
            commit_hash=snapshot['commit_hash'],
            model_version=snapshot['model_hash'][:8]
        )
        
        if success:
            logger.info("Successfully restored pre-tuning state")
        else:
            logger.critical("Failed to restore system state after tuning failure!")

class InfrastructureManager:
    """Orchestrates core infrastructure operations"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.tuner = HyperparamTuner(**self.config)
        self.rollback = RollbackSystem(self.config)

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required sections
        required_sections = ['rollback', 'paths']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config

    def automated_tuning_cycle(self):
        """Complete tuning cycle with safety mechanisms"""
        try:
            return self.tuner.run_safe_tuning()
        except Exception as e:
            logger.error(f"Automated tuning cycle failed: {str(e)}")
            return None

    def emergency_rollback(self):
        """Restore to last known good configuration"""
        last_stable = self._get_last_stable_version()
        return self.rollback.full_rollback(
            commit_hash=last_stable['commit'],
            model_version=last_stable['model']
        )

    def _get_last_stable_version(self) -> Dict[str, str]:
        """Retrieve last validated system state by checking git and model backups."""
        import subprocess
        from pathlib import Path
        from src.deployment.rollback.model_rollback import validate_backup_integrity
    
        try:
            # 1. Get previous Git commit (one before HEAD)
            prev_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD~1']).decode().strip()
    
            # 2. Get the latest model backup available
            backup_dir = Path(self.config['rollback']['backup_dir'])
            backups = sorted(backup_dir.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    
            if not backups:
                raise FileNotFoundError("No model backups found!")
    
            latest_backup = backups[0]
            model_version = latest_backup.stem  # Example: "v1.4.2"
    
            # 3. Optionally verify integrity
            if not validate_backup_integrity(latest_backup, latest_backup):
                raise ValueError("Latest model backup integrity check failed.")
    
            return {
                'commit': prev_commit,
                'model': model_version
            }
    
        except Exception as e:
            logger.error(f"Failed to retrieve last stable version: {str(e)}")
            # As fallback: assume current commit and no model rollback
            current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            return {
                'commit': current_commit,
                'model': 'unknown'
            }
