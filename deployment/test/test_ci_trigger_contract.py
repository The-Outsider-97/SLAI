import pytest
import urllib.error

import deployment.ci_trigger as ci_trigger


class DummyResponse:
    def __init__(self, status_code, text=""):
        self.status = status_code
        self._text = text

    def read(self):
        return self._text.encode("utf-8")


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_trigger_success_dispatches_github_api(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN_DEV", "token")
    session = DummySession([DummyResponse(204)])
    connector = ci_trigger.GitHubConnector("dev", http_client=session)

    assert connector.trigger("develop") is True
    assert session.calls == 1


def test_trigger_fails_without_auth_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN_DEV", raising=False)
    connector = ci_trigger.GitHubConnector("dev", http_client=DummySession([DummyResponse(204)]))

    with pytest.raises(EnvironmentError):
        connector.trigger("develop")


def test_trigger_retries_transient_http_then_succeeds(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN_DEV", "token")
    session = DummySession([DummyResponse(503, "busy"), DummyResponse(204)])
    connector = ci_trigger.GitHubConnector("dev", http_client=session, retry_delay_seconds=0)

    assert connector.trigger("develop") is True
    assert session.calls == 2


def test_trigger_retries_request_exception_then_fails(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN_DEV", "token")
    session = DummySession(
        [
            urllib.error.URLError("net1"),
            urllib.error.URLError("net2"),
            urllib.error.URLError("net3"),
        ]
    )
    connector = ci_trigger.GitHubConnector("dev", http_client=session, retry_delay_seconds=0)

    with pytest.raises(ConnectionError):
        connector.trigger("develop")
    assert session.calls == 3


def test_trigger_rejects_invalid_branch(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN_DEV", "token")
    connector = ci_trigger.GitHubConnector("dev", http_client=DummySession([DummyResponse(204)]))

    with pytest.raises(ValueError):
        connector.trigger("bad branch")


def test_trigger_ci_single_authoritative_path(monkeypatch):
    class FakeConnector:
        def __init__(self):
            self.called = False

        def trigger(self, branch):
            self.called = True
            assert branch == "main"
            return True

    fake = FakeConnector()
    monkeypatch.setattr(ci_trigger, "get_connector", lambda env: fake)

    assert ci_trigger.trigger_ci(env="prod", branch="main") is True
    assert fake.called is True
