import time
import urllib.error
import urllib.request

from deployment.audit_logger import log_event
from deployment.ci_trigger import trigger_ci
from deployment.release_invariants import ensure_release_invariants
from deployment.rollback import rollback_model
from deployment.rollback.code_rollback import rollback_to_previous_tag
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Deployment Manager")
printer = PrettyPrinter

def _memory_usage_percent() -> float:
    """Best-effort memory usage check without optional third-party dependencies."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            mem = {}
            for line in f:
                key, value = line.split(":", 1)
                mem[key.strip()] = int(value.strip().split()[0])
        total = mem.get("MemTotal", 0)
        available = mem.get("MemAvailable", 0)
        if total <= 0:
            return 0.0
        used = total - available
        return (used / total) * 100.0
    except OSError:
        return 0.0


class DeploymentManager:
    def __init__(self, env="prod"):
        self.env = env

    def _pre_deployment_check(self):
        """Precondition checks before any deployment dispatch."""
        if _memory_usage_percent() > 90:
            raise RuntimeError("Insufficient memory for deployment")

        ensure_release_invariants(strict=True)

    def _post_deployment_verify(self):
        """Post-deploy health verification with retry budget."""
        health_url = f"https://{self.env}.example.com/health"
        for _ in range(3):
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        raise ConnectionError("Service health check failed after deployment")

    def _deploy_to_canary_cluster(self, branch: str):
        """
        Canary rollout placeholder.

        This function intentionally behaves as a no-op hook until real canary
        infrastructure is wired, while preserving a stable deployment contract.
        """
        logger.info("Canary rollout hook invoked for branch=%s env=%s", branch, self.env)

    def deploy(self, user: str, branch: str, version: str = None):
        logger.info("Starting deployment: env=%s branch=%s", self.env, branch)
        try:
            self._pre_deployment_check()

            if self.env == "prod":
                self._deploy_to_canary_cluster(branch)
                time.sleep(1)

            trigger_ci(env=self.env, branch=branch)
            self._post_deployment_verify()
            log_event(
                event_type="deploy",
                user=user,
                environment=self.env,
                branch=branch,
                version=version,
                success=True,
                details={"env": self.env},
            )

        except Exception as e:
            logger.error("Deployment failed: %s", e)
            log_event(
                event_type="deploy",
                user=user,
                environment=self.env,
                branch=branch,
                version=version,
                success=False,
                details={"env": self.env, "error": str(e)},
            )
            raise

    def rollback(self, user: str, reason: str, rollback_code=True, rollback_model_flag=True):
        logger.warning("Initiating rollback: env=%s reason=%s", self.env, reason)
        try:
            if rollback_model_flag:
                rollback_model(models_dir=f"models/{self.env}/", backup_dir=f"models/{self.env}/backups/")
            if rollback_code:
                rollback_to_previous_tag()
            log_event(
                event_type="rollback",
                user=user,
                environment=self.env,
                branch="unknown",
                success=True,
                details={"reason": reason, "env": self.env},
            )
        except Exception as e:
            logger.error("Rollback failed: %s", e)
            log_event(
                event_type="rollback",
                user=user,
                environment=self.env,
                branch="unknown",
                success=False,
                details={"reason": reason, "env": self.env, "error": str(e)},
            )
            raise
