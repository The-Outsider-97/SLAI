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

    # Model rollback logic stays...

    def git_reset_to_commit(self, commit_hash: str, hard_reset: bool = True):
        cmd = ['git', 'reset', '--hard' if hard_reset else '--soft', commit_hash]
        subprocess.run(cmd, check=True)
        print(f" Git reset to {commit_hash}")

    def git_delete_tag(self, tag: str, remote: bool = True):
        subprocess.run(['git', 'tag', '-d', tag], check=True)
        print(f" Deleted local tag {tag}")
        if remote:
            subprocess.run(['git', 'push', 'origin', f':refs/tags/{tag}'], check=True)
            print(f" Deleted remote tag {tag}")

    def git_rollback_to_previous_release(self):
        tags = subprocess.run(['git', 'tag', '--sort=-creatordate'], stdout=subprocess.PIPE, check=True)
        tag_list = tags.stdout.decode().splitlines()

        if len(tag_list) < 2:
            print(" No previous tag to rollback to.")
            return

        latest_tag = tag_list[0]
        previous_tag = tag_list[1]

        prev_commit = subprocess.run(['git', 'rev-list', '-n', '1', previous_tag], stdout=subprocess.PIPE, check=True)
        prev_commit_hash = prev_commit.stdout.decode().strip()

        print(f"Rolling back to tag {previous_tag} at commit {prev_commit_hash}")
        self.git_reset_to_commit(prev_commit_hash)
        self.git_delete_tag(latest_tag)

    def trigger_action(self, reason):
        print(f"Initiating action due to: {reason}")

        if self.rollback_handler:
            print("Rolling back model artifacts...")
            self.rollback_handler.rollback_model()

        if self.git_rollback_handler:
            print("Rolling back code repository...")
            self.git_rollback_handler.git_rollback_to_previous_release()

        if self.hyperparam_tuner:
            print("Triggering hyperparameter tuning and retraining...")
            self.hyperparam_tuner.run_tuning_pipeline()


    except subprocess.CalledProcessError as e:
        logger.error(f"Rollback failed: {e}")
