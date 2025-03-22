import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Placeholder CI/CD endpoints
CI_CONFIG = {
    "github": {
        "repo": "The-Outsider-97/SLAI",
        "token": "your_github_pat_here"
    },
    "gitlab": {
        "project_id": "123456",
        "token": "your_gitlab_token_here"
    },
    "jenkins": {
        "base_url": "http://jenkins.local:8080",
        "job_name": "SLAI_Build",
        "token": "jenkins_token_here"
    }
}


def trigger_github_actions(branch="main"):
    """Trigger a GitHub Actions workflow_dispatch."""
    url = f"https://api.github.com/repos/{CI_CONFIG['github']['repo']}/actions/workflows/main.yml/dispatches"
    headers = {
        "Authorization": f"token {CI_CONFIG['github']['token']}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "ref": branch
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 204:
        logger.info(f"GitHub Actions triggered successfully for branch '{branch}'")
    else:
        logger.error(f"GitHub Actions trigger failed: {response.status_code} - {response.text}")


def trigger_gitlab_pipeline(branch="main"):
    """Trigger GitLab CI/CD pipeline."""
    url = f"https://gitlab.com/api/v4/projects/{CI_CONFIG['gitlab']['project_id']}/trigger/pipeline"
    payload = {
        "token": CI_CONFIG['gitlab']['token'],
        "ref": branch
    }

    response = requests.post(url, data=payload)
    if response.ok:
        logger.info(f"GitLab CI/CD pipeline triggered for branch '{branch}'")
    else:
        logger.error(f"GitLab pipeline trigger failed: {response.status_code} - {response.text}")


def trigger_jenkins_build():
    """Trigger Jenkins build via remote API."""
    base = CI_CONFIG['jenkins']['base_url']
    job = CI_CONFIG['jenkins']['job_name']
    token = CI_CONFIG['jenkins']['token']

    url = f"{base}/job/{job}/build?token={token}"
    response = requests.post(url)

    if response.ok:
        logger.info(f"Jenkins job '{job}' triggered successfully.")
    else:
        logger.error(f"Jenkins job trigger failed: {response.status_code} - {response.text}")
