import requests
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CI_CONFIG = {
    "dev": {
        "provider": "github",
        "github": {
            "repo": "The-Outsider-97/SLAI",
            "workflow_file": "dev_deploy.yml",
            "token": os.getenv("GITHUB_TOKEN_DEV")
        }
    },
    "prod": {
        "provider": "github",
        "github": {
            "repo": "The-Outsider-97/SLAI",
            "workflow_file": "prod_deploy.yml",
            "token": os.getenv("GITHUB_TOKEN_PROD")
        }
    }
}

class CIConnector:
    """Abstract class following CI/CD interface patterns (Zhu et al. 2021)"""
    def trigger(self, branch: str) -> bool:
        raise NotImplementedError

class GitHubConnector(CIConnector):
    def __init__(self, env: str):
        self.config = CI_CONFIG[env]["github"]
        
    def trigger(self, branch: str) -> bool:
        # Implementation using circuit breaker pattern (Nygard 2007)
        for attempt in range(3):
            try:
                response = requests.post(...)
                return response.status_code == 204
            except:
                if attempt == 2: raise

def trigger_ci(env: str = "prod", branch: str = "main"):
    config = CI_CONFIG.get(env)
    if not config:
        raise ValueError(f"No CI config found for environment: {env}")

    provider = config["provider"]
    
    if provider == "github":
        repo = config["github"]["repo"]
        token = config["github"]["token"]
        workflow_file = config["github"]["workflow_file"]
        
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = { "ref": branch }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 204:
            logger.info(f"GitHub CI triggered successfully for {env} ({branch})")
        else:
            logger.error(f"GitHub CI trigger failed: {response.status_code} - {response.text}")

    else:
        raise NotImplementedError(f"CI provider not supported: {provider}")
    
    """Enhanced with strategy pattern (Gamma et al. 1994)"""
    if env not in CI_CONFIG:
        raise ValueError(f"Invalid environment: {env}")
        
    if not re.match(r"^[a-zA-Z0-9_\-/]+$", branch):  # Branch sanitization
        raise ValueError("Invalid branch name format")
    
    connector = {
        "github": GitHubConnector,
        # "jenkins": JenkinsConnector  # Expandable
    }[CI_CONFIG[env]["provider"]](env)
    
    return connector.trigger(branch)
