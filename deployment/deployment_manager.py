import requests
import logging
import os, sys
import psutil
import time

from rollback import rollback_model
from rollback.code_rollback import rollback_to_previous_tag
from deployment.ci_trigger import trigger_ci
from deployment.audit_logger import log_event

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DeploymentManager:
    def __init__(self, env="prod"):
        self.env = env

    def _pre_deployment_check(self):
        """Academic-inspired precondition checks (Lee et al. 2022 - Safe Deployment Patterns)"""
        # 1. Resource availability check
        if psutil.virtual_memory().percent > 90:
            raise RuntimeError("Insufficient memory for deployment")
        
        # 2. Dependency consistency verification
        if not os.path.exists("requirements.lock"):
            raise FileNotFoundError("Missing version-locked dependencies file")

    def _post_deployment_verify(self):
        """Phase-based verification (Khomh et al. 2015 - Rolling Updates)"""
        # 1. Service health check
        health_url = f"https://{self.env}.example.com/health"
        for _ in range(3):  # Retry 3 times
            try:
                if requests.get(health_url, timeout=5).status_code == 200:
                    return
            except:
                time.sleep(2)
        raise ConnectionError("Service health check failed after deployment")

    def deploy(self, user: str, branch: str, version: str = None):
        logger.info(f"Starting deployment: {self.env=} {branch=}")
        try:
            self._pre_deployment_check()
            # Staggered rollout (Nakajima et al. 2020 - Phased Deployment)
            if self.env == "prod":
                self._deploy_to_canary_cluster(branch)
                time.sleep(300)  # 5min observation period
            
            trigger_ci(env=self.env, branch=branch)
            self._post_deployment_verify()
            log_event(event_type="deploy", user=user, branch=branch, version=version, success=True,
                      details={"env": self.env})
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            log_event(event_type="deploy", user=user, branch=branch, version=version, success=False,
                      details={"env": self.env, "error": str(e)})

    def rollback(self, user: str, reason: str, rollback_code=True, rollback_model_flag=True):
        logger.warning(f"Initiating rollback: {self.env=} reason={reason}")
        try:
            if rollback_model_flag:
                rollback_model(models_dir=f"models/{self.env}/", backup_dir=f"models/{self.env}/backups/")
            if rollback_code:
                rollback_to_previous_tag()
            log_event(event_type="rollback", user=user, branch="unknown", success=True,
                      details={"reason": reason, "env": self.env})
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            log_event(event_type="rollback", user=user, branch="unknown", success=False,
                      details={"reason": reason, "env": self.env, "error": str(e)})
