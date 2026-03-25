import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("CI Trigger")
printer = PrettyPrinter

BRANCH_PATTERN = re.compile(r"^[a-zA-Z0-9_./\-]+$")


CI_CONFIG: Dict[str, Dict[str, Any]] = {
    "dev": {
        "provider": "github",
        "github": {
            "repo": "The-Outsider-97/SLAI",
            "workflow_file": "dev_deploy.yml",
            "token_env": "GITHUB_TOKEN_DEV",
        },
    },
    "prod": {
        "provider": "github",
        "github": {
            "repo": "The-Outsider-97/SLAI",
            "workflow_file": "prod_deploy.yml",
            "token_env": "GITHUB_TOKEN_PROD",
        },
    },
}


class HttpResponse(Protocol):
    status: int

    def read(self) -> bytes: ...


class CIConnector:
    def trigger(self, branch: str) -> bool:
        raise NotImplementedError


@dataclass
class GitHubDispatchRequest:
    repo: str
    workflow_file: str
    branch: str

    @property
    def dispatch_url(self) -> str:
        return f"https://api.github.com/repos/{self.repo}/actions/workflows/{self.workflow_file}/dispatches"

    @property
    def payload(self) -> bytes:
        return json.dumps({"ref": self.branch}).encode("utf-8")


class UrllibHttpClient:
    def post(self, url: str, headers: Dict[str, str], payload: bytes, timeout: int) -> HttpResponse:
        req = urllib.request.Request(url=url, data=payload, headers=headers, method="POST")
        return urllib.request.urlopen(req, timeout=timeout)


class GitHubConnector(CIConnector):
    def __init__(
        self,
        env: str,
        http_client: Optional[UrllibHttpClient] = None,
        timeout_seconds: int = 15,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        if env not in CI_CONFIG:
            raise ValueError(f"Invalid environment: {env}")

        cfg = CI_CONFIG[env]["github"]
        self.repo = cfg["repo"]
        self.workflow_file = cfg["workflow_file"]
        self.token_env = cfg["token_env"]

        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.http_client = http_client or UrllibHttpClient()

    def _build_headers(self) -> Dict[str, str]:
        token = os.getenv(self.token_env)
        if not token:
            raise EnvironmentError(f"Missing GitHub token in environment variable: {self.token_env}")

        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        }

    def trigger(self, branch: str) -> bool:
        if not BRANCH_PATTERN.match(branch):
            raise ValueError("Invalid branch name format")

        req = GitHubDispatchRequest(self.repo, self.workflow_file, branch)
        headers = self._build_headers()

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.http_client.post(
                    req.dispatch_url,
                    headers=headers,
                    payload=req.payload,
                    timeout=self.timeout_seconds,
                )
                status_code = response.status
                body = response.read().decode("utf-8", errors="ignore")

                if status_code == 204:
                    logger.info("GitHub CI triggered successfully: workflow=%s branch=%s", self.workflow_file, branch)
                    return True

                if status_code in {401, 403}:
                    raise PermissionError(f"GitHub dispatch authorization failed ({status_code}): {body}")

                if status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    logger.warning(
                        "Transient CI dispatch failure (attempt=%s/%s, status=%s). Retrying...",
                        attempt,
                        self.max_retries,
                        status_code,
                    )
                    time.sleep(self.retry_delay_seconds * attempt)
                    continue

                raise RuntimeError(f"GitHub CI trigger failed with status {status_code}: {body}")
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in {401, 403}:
                    raise PermissionError(f"GitHub dispatch authorization failed ({exc.code})") from exc
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds * attempt)
                    continue
                raise RuntimeError(f"GitHub CI trigger failed with status {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_delay_seconds * attempt)

        raise ConnectionError("Failed to dispatch CI workflow after retry budget") from last_exc


def get_connector(env: str) -> CIConnector:
    if env not in CI_CONFIG:
        raise ValueError(f"No CI config found for environment: {env}")

    provider = CI_CONFIG[env]["provider"]
    connector_map = {
        "github": GitHubConnector,
    }

    if provider not in connector_map:
        raise NotImplementedError(f"CI provider not supported: {provider}")

    return connector_map[provider](env)


def trigger_ci(env: str = "prod", branch: str = "main") -> bool:
    """Single authoritative CI trigger path."""
    connector = get_connector(env)
    return connector.trigger(branch)


if __name__ == "__main__":
    print("\n=== Running CI Trigger ===\n")
    printer.status("TEST", "Starting CI Trigger tests", "info")

    # connector = GitHubConnector(
    #     env="prod",
    #     http_client=None,
    #     timeout_seconds=15
    # )

    # Choose environment and branch
    env = "prod"          # or "dev"
    branch = "main"

    try:
        success = trigger_ci(env=env, branch=branch)
        if success:
            print(f"✅ CI triggered successfully on {env} with branch {branch}")
        else:
            print(f"❌ CI trigger failed on {env}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n=== CI Trigger Test Complete ===")
