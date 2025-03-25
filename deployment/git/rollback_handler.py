import os
import shutil
import subprocess
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RollbackHandler:
    """
    Handles both model rollback (file system) and Git rollback (source control).
    """
    def __init__(self, models_dir='models/', backup_dir='models/backups/'):
        self.models_dir = models_dir
        self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)

    def rollback_model(self):
        """
        Rolls back model files from the backup directory.
        """
        try:
            logger.info("Rolling back model from backup...")

            if not os.path.exists(self.backup_dir):
                logger.error(f"Backup directory does not exist: {self.backup_dir}")
                return

            # Safely delete everything in models/ EXCEPT backups/
            if os.path.exists(self.models_dir):
                for item in os.listdir(self.models_dir):
                    item_path = os.path.join(self.models_dir, item)
                    if item != 'backups':
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)

            # Restore backup to models directory
            for item in os.listdir(self.backup_dir):
                s = os.path.join(self.backup_dir, item)
                d = os.path.join(self.models_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)

        except Exception as e:
            logger.error(f"Model rollback failed: {e}")

    def git_reset_to_commit(self, commit_hash: str, hard_reset: bool = True):
        cmd = ['git', 'reset', '--hard' if hard_reset else '--soft', commit_hash]
        subprocess.run(cmd, check=True)
        print(f"Git reset to {commit_hash}")

    def git_delete_tag(self, tag: str, remote: bool = True):
        subprocess.run(['git', 'tag', '-d', tag], check=True)
        print(f"Deleted local tag {tag}")
        if remote:
            subprocess.run(['git', 'push', 'origin', f':refs/tags/{tag}'], check=True)
            print(f"Deleted remote tag {tag}")

    def git_rollback_to_previous_release(self):
        tags = subprocess.run(['git', 'tag', '--sort=-creatordate'], stdout=subprocess.PIPE, check=True)
        tag_list = tags.stdout.decode().splitlines()

        if len(tag_list) < 2:
            print("No previous tag to rollback to.")
            return

        latest_tag = tag_list[0]
        previous_tag = tag_list[1]

        prev_commit = subprocess.run(['git', 'rev-list', '-n', '1', previous_tag], stdout=subprocess.PIPE, check=True)
        prev_commit_hash = prev_commit.stdout.decode().strip()

        print(f"Rolling back to tag {previous_tag} at commit {prev_commit_hash}")
        self.git_reset_to_commit(prev_commit_hash)
        self.git_delete_tag(latest_tag)

    def trigger_action(self, reason, git_rollback=False, hyperparam_tuner=None):
        print(f"Initiating rollback due to: {reason}")

        print("Rolling back model artifacts...")
        self.rollback_model()

        if git_rollback:
            print("Rolling back code repository...")
            self.git_rollback_to_previous_release()

        if hyperparam_tuner:
            print("Triggering hyperparameter tuning and retraining...")
            hyperparam_tuner.run_tuning_pipeline()
