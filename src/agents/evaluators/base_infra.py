"""
BASE_INFRA.PY - Core Infrastructure Components
Implements version control, rollback, and hyperparameter tuning foundations
"""

import subprocess
import hashlib
import json, yaml
import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from deployment.rollback.code_rollback import AtomicRollback, reset_to_commit, rollforward_to_next_tag, get_sorted_tags
from deployment.rollback.model_rollback import rollback_model, validate_backup_integrity, rollforward_model
from src.tuning.tuner import HyperparamTuner as BaseTuner
from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger

logger = get_logger("Infrastructure Manager")

CONFIG_PATH = "src/agents/evaluators/configs/evaluator_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class InfrastructureManager:
    """Orchestrates core infrastructure operations"""
    
    def __init__(self, config_path=CONFIG_PATH):
        self.full_config = load_config(config_path)
        # Extract infrastructure manager specific config
        self.config = self.full_config.get('infrastructure_manager', {})
        
        # Initialize components with proper configuration
        self.memory = EvaluatorsMemory(self.full_config)
        self.rollback = RollbackSystem(self.config)
        self.tuner = self._init_tuner()

        logger.info("Infrastructure Manager initialized with %s strategy",
                   self.config.get('tuning_strategy', 'bayesian'))

    def _init_tuner(self):
        """Initialize tuner using infrastructure manager config"""
        return HyperparamTuner(
            evaluation_function=self._agent_evaluator,
            strategy=self.config.get('tuning_strategy', 'bayesian'),
            n_calls=self.config.get('n_calls', 20),
            n_random_starts=self.config.get('n_random_starts', 5),
            config_path="src/tuning/configs/hyperparam.yaml"
        )

    def _agent_evaluator(self, params):
        """Integrated evaluation function using agent's metrics"""
        # Actual implementation would use agent's performance metrics
        risk_score = self.config['risk_adaptation'].get('initial_hazard_rates', 0.2)
        return risk_score * np.random.rand()  # Placeholder

    def full_tuning_cycle(self):
        """Complete tuning cycle with integrated risk management"""
        try:
            result = self.tuner.run_safe_tuning()
            self._update_risk_profile(result)
            return result
        except Exception as e:
            logger.error("Tuning cycle failed: %s", str(e))
            self.rollback.full_rollback()
            raise

    def _update_risk_profile(self, tuning_result):
        """Update risk adaptation parameters based on tuning results"""
        new_rate = tuning_result.get('score', 0.2) * 0.95
        self.config['risk_adaptation']['initial_hazard_rates'] = new_rate
        logger.info("Updated risk profile to %.2f", new_rate)

    def automated_tuning_cycle(self):
        """Complete tuning cycle with safety mechanisms"""
        try:
            return self.tuner.run_safe_tuning()
        except Exception as e:
            logger.error(f"Automated tuning cycle failed: {str(e)}")
            return None

    def emergency_rollback(self):
        """Safe rollback entry point"""
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode()
        if status.strip():
            logger.error("Aborting rollback: Uncommitted changes detected")
            raise RuntimeError("Commit changes before rollback")
        
        return self.rollback.full_rollback()

    def _get_last_stable_version(self) -> Dict[str, str]:
        """Retrieve last validated system state by checking git and model backups."""
        result = subprocess.check_output(['git', 'describe', '--abbrev=0']).decode().strip()
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

            return {'commit': result, 'model': result.lstrip('v')}

        except Exception as e:
            logger.error(f"Failed to retrieve last stable version: {str(e)}")
            # As fallback: assume current commit and no model rollback
            current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            return {
                'commit': current_commit,
                'model': 'unknown'
            }

    def rollforward(self):
        """Roll forward to a newer system version if available"""
        try:
            return rollforward_model()
        except Exception as e:
            logger.error(f"Rollforward failed: {str(e)}")
            return False

    def rollforward_code(self):
        """Roll forward to a newer code version if available"""
        try:
            status = subprocess.check_output(['git', 'status', '--porcelain']).decode()
            if status.strip():
                logger.error("Aborting rollforward: Uncommitted changes detected")
                raise RuntimeError("Commit changes before rollforward")
            
            return rollforward_to_next_tag()
        except Exception as e:
            logger.error(f"Code rollforward failed: {str(e)}")
            return False

class RollbackSystem:
    """Implements atomic rollback operations across code and model versions"""
    
    def __init__(self, config):
        self.backup_dir = Path(config.get('backup_dir', 'models/backups/'))
        self.model_dir = Path(config.get('model_dir', 'models/'))
        self._verify_directories()

    def _verify_directories(self):
        """Ensure required directories exist with proper permissions"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def full_rollback(self):
        """Interactive rollback with confirmation and version selection"""
        print("\n=== ROLLBACK WARNING ===")
        print("This will revert both code and model to previous versions.")
        print("Any uncommitted changes will be lost!\n")
        
        confirm = input("Are you absolutely sure? (type 'confirm' to proceed): ").lower()
        if confirm != 'confirm':
            logger.info("Rollback aborted by user")
            return False

        # Get available versions
        code_versions = self.get_available_code_versions()
        model_versions = self.get_available_model_versions()

        if not code_versions or not model_versions:
            logger.error("No rollback targets available")
            return False

        # Display code versions
        print("\nAvailable code versions:")
        for idx, (tag, ts) in enumerate(code_versions[:10]):
            print(f"{idx+1}. {tag} ({ts})")

        # Get user selection
        try:
            code_choice = int(input("\nEnter code version number: "))-1
            selected_code = code_versions[code_choice][0]
            
            print("\nAvailable model versions:")
            for idx, (name, ts) in enumerate(model_versions[:10]):
                print(f"{idx+1}. {name} ({ts.strftime('%Y-%m-%d %H:%M')})")
            
            model_choice = int(input("\nEnter model version number: "))-1
            selected_model = model_versions[model_choice][0]
        except (ValueError, IndexError):
            logger.error("Invalid selection")
            return False

        # Execute rollback
        return self._commit_rollbacks(selected_code, selected_model)

    def get_available_code_versions(self):
        
        return [(tag, ts.isoformat()) for tag, ts in get_sorted_tags()]

    def get_available_model_versions(self):
        backups = sorted(
            [d for d in self.backup_dir.iterdir() if d.is_dir() and d.name.startswith("rollback_")],
            key=lambda x: x.name, 
            reverse=True
        )
        return [
            (b.name, datetime.strptime(b.name.split("_")[1], "%Y%m%d_%H%M%S")) 
            for b in backups
        ]

    def _create_backup(self):
        pass

        try:
            # Check for uncommitted changes
            status = subprocess.check_output(['git', 'status', '--porcelain']).decode()
            if status.strip():
                logger.error("Aborting rollback: Uncommitted changes detected")
                raise RuntimeError("Commit changes before rollback")
            
            subprocess.run(['git', 'checkout', commit_hash, '--', '.'], check=True)
            logger.info(f"Checked out files from {commit_hash}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False

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

    def rollforward_code(self):
        try:
            status = subprocess.check_output(['git', 'status', '--porcelain']).decode()
            if status.strip():
                logger.info("Stashing uncommitted changes temporarily...")
                subprocess.run(['git', 'stash', '--include-untracked'], check=True)
                
            success = rollforward_to_next_tag()
            
            if status.strip():
                subprocess.run(['git', 'stash', 'pop'], check=True)
            return success
        except Exception as e:
            logger.error(f"Code rollforward failed: {str(e)}")
            return False

class HyperparamTuner(BaseTuner):
    """Extended tuner with rollback integration"""
    def __init__(self, evaluation_function, strategy='bayesian', 
                 n_calls=20, n_random_starts=5, **kwargs):
        super().__init__(
            evaluation_function=evaluation_function,
            strategy=strategy,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            config_path=self._get_config_path(strategy),
            allow_generate=True
        )
        self.rollback = RollbackSystem(kwargs.get('rollback_config', {}))
        self.best_state = None

    def _get_config_path(self, strategy):
        """Resolve config path based on strategy"""
        return f"src/tuning/configs/generated_config_{strategy}.json"

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

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Adaptive Risk ===\n")

    infra = InfrastructureManager()
    
    try:
        print("\nAttempting code rollforward...")
        if infra.rollforward_code():
            print("✅ Successfully rolled forward code version")
            
        print("\nAttempting model rollforward...")
        if infra.rollforward():
            print("✅ Successfully rolled forward model version")
            
    except Exception as e:
        print(f"❌ Rollforward failed: {str(e)}")
    
    print(f"\n* * * * * Phase 2 * * * * *\n")

    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Adaptive Risk ===\n")
