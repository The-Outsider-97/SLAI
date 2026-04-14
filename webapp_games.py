"""Application entry point for R-Games.

Responsibilities:
1. Host the shared backend server.
2. Initialize game AI runtimes on-demand.
3. Route AI API calls for the selected game within each user session.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys, os
import threading
import uuid
import requests

from dataclasses import dataclass
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
SESSION_COOKIE = "r_games_session"

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.planning.planning_types import Task, TaskType
from logs.logger import get_logger, PrettyPrinter


logger = get_logger("R-Games")
printer = PrettyPrinter()

@dataclass(frozen=True)
class GameConfig:
    key: str
    display_name: str
    ai_module: str
    ai_initializer: str
    launch_url: str


GAMES: dict[str, GameConfig] = {
    "chronos": GameConfig(
        key="chronos",
        display_name="Chronos",
        ai_module="games.ai_chronos",
        ai_initializer="initialize_ai",
        launch_url="/games/chronos/index.html",
    ),
    "aether_shift": GameConfig(
        key="aether_shift",
        display_name="Aether Shift",
        ai_module="games.ai_aether",
        ai_initializer="initialize_ai",
        launch_url="/games/aether/index.html",
    ),
    "mindweave": GameConfig(
        key="mindweave",
        display_name="Project: Mindweave",
        ai_module="games.ai_mindweave",
        ai_initializer="initialize_ai",
        launch_url="/games/mindweave/index.html",
    ),
    "patolli": GameConfig(
        key="patolli",
        display_name="Patolli",
        ai_module="games.ai_patolli",
        ai_initializer="initialize_ai",
        launch_url="/games/Patolli/index.html",
    ),
    "puluc": GameConfig(
        key="puluc",
        display_name="Puluc",
        ai_module="games.ai_puluc",
        ai_initializer="initialize_ai",
        launch_url="/games/Puluc/index.html",
    ),
    "parallax": GameConfig(
        key="parallax",
        display_name="Parallax Protocol",
        ai_module="games.ai_parallax",
        ai_initializer="initialize_ai",
        launch_url="/games/parallax/index.html",
    ),
}


class GameRuntime:
    """Owns game AI instances and session-to-game selections."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.ai_instances: dict[str, object] = {}
        self.session_selected_game: dict[str, str] = {}
        self._module_status_cache: dict[str, dict[str, Any]] = {}

    def _load_initializer(self, config: GameConfig) -> Callable[[], object]:
        module = importlib.import_module(config.ai_module)
        initializer = getattr(module, config.ai_initializer, None)
        if not callable(initializer):
            raise AttributeError(
                f"{config.ai_module}.{config.ai_initializer} is missing or not callable"
            )
        return initializer

    def _module_status(self, config: GameConfig) -> dict[str, Any]:
        cached = self._module_status_cache.get(config.key)
        if cached is not None:
            return cached

        try:
            self._load_initializer(config)
            status = {"available": True, "reason": None}
        except Exception as error:  # noqa: BLE001
            status = {"available": False, "reason": str(error)}

        self._module_status_cache[config.key] = status
        return status

    def list_games(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "key": game.key,
                    "name": game.display_name,
                    "launch_url": game.launch_url,
                    **self._module_status(game),
                    "initialized": game.key in self.ai_instances,
                }
                for game in GAMES.values()
            ]

    def select_game(self, session_id: str, game_key: str) -> dict[str, str]:
        config = GAMES.get(game_key)
        if config is None:
            raise KeyError(f"Unknown game '{game_key}'")

        with self._lock:
            status = self._module_status(config)
            if not status["available"]:
                raise RuntimeError(
                    f"{config.display_name} unavailable: {status['reason']}"
                )

            if game_key not in self.ai_instances:
                logger.info("Initializing AI runtime for %s", config.display_name)
                initializer = self._load_initializer(config)
                self.ai_instances[game_key] = initializer()

            self.session_selected_game[session_id] = game_key

        return {
            "game": config.key,
            "name": config.display_name,
            "launch_url": config.launch_url,
            "message": f"{config.display_name} initialized. Launching now...",
        }

    def get_selected_game(self, session_id: str) -> str | None:
        return self.session_selected_game.get(session_id)

    def _get_selected_instance(self, session_id: str) -> tuple[GameConfig, object]:
        game_key = self.get_selected_game(session_id)
        if not game_key:
            raise LookupError("No game selected for this session")

        config = GAMES[game_key]
        instance = self.ai_instances.get(game_key)
        if instance is None:
            raise LookupError(f"AI for {config.display_name} is not initialized")

        return config, instance
    
    def _infer_game_from_referer(self, referer: str | None) -> str | None:
        """Map a Referer URL to a game key, e.g. '/games/aether/index.html' -> 'aether_shift'."""
        if not referer:
            return None
        # Normalise path: remove query and fragment
        path = referer.split('?')[0].split('#')[0]
        for game in GAMES.values():
            if game.launch_url in path:
                return game.key
        return None

    def ai_health(self, session_id: str) -> dict[str, Any]:
        game_key = self.get_selected_game(session_id)
        if not game_key:
            return {"status": "idle", "selected_game": None}

        config = GAMES[game_key]
        instance = self.ai_instances.get(game_key)
        if instance is None:
            return {
                "status": "initializing",
                "selected_game": game_key,
                "name": config.display_name,
            }

        status = {"status": "ready", "selected_game": game_key, "name": config.display_name}
        if hasattr(instance, "health") and callable(instance.health):
            custom = instance.health()
            if isinstance(custom, dict):
                status.update(custom)
        return status

    def ai_move(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            game_key = self.get_selected_game(session_id)
            if game_key is None:
                # Auto-select from Referer
                game_key = self._infer_game_from_referer(referer)
                if game_key is None:
                    raise LookupError("No game selected for this session and could not infer from Referer")
                # Perform selection (initialises AI if needed)
                self.select_game(session_id, game_key)
                logger.info("Auto-selected game %s for session %s based on Referer", game_key, session_id)
            config = GAMES[game_key]
            instance = self.ai_instances.get(game_key)
            if instance is None:
                raise LookupError(f"AI for {config.display_name} is not initialised")

        _, instance = self._get_selected_instance(session_id)

        move_fn = getattr(instance, "get_move", None)
        if not callable(move_fn):
            raise NotImplementedError("Selected game AI does not implement get_move")

        move = move_fn(payload)
        if move is None:
            raise LookupError("No valid move found")

        if isinstance(move, dict) and "choice" in move:
            return {"choice": move["choice"]}

        return {"move": move}

    def ai_learn(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _, instance = self._get_selected_instance(session_id)
        learn_fn = getattr(instance, "learn_from_game", None)
        if not callable(learn_fn):
            return {
                "status": "ignored",
                "message": "Selected game AI does not implement learning updates",
            }

        success = learn_fn(payload)
        if success:
            return {"status": "success", "message": "Learning updated"}

        return {"status": "error", "message": "Learning update failed"}

    def ai_chat(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _, instance = self._get_selected_instance(session_id)
        chat_fn = getattr(instance, "chat", None)
        if not callable(chat_fn):
            raise NotImplementedError("Selected game AI does not implement chat")

        result = chat_fn(payload)
        if not isinstance(result, dict):
            raise TypeError("Chat response must be a JSON object")
        return result

runtime = GameRuntime()

def _infer_llm_provider(api_key: str) -> str:
    """Infer provider from common API key prefixes."""
    key = api_key.strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("sk-or-v1"):
        return "openrouter"
    if key.startswith("xai-"):
        return "xai"
    if key.startswith("AIza"):
        return "gemini"
    return "openai"


def _build_validation_request(provider: str, api_key: str) -> tuple[str, dict[str, str]]:
    """Build provider-specific validation URL and headers."""
    key = api_key.strip()
    if provider == "anthropic":
        return (
            os.getenv("MINDWEAVE_ANTHROPIC_VALIDATION_URL", "https://api.anthropic.com/v1/models"),
            {
                "x-api-key": key,
                "anthropic-version": os.getenv("MINDWEAVE_ANTHROPIC_VERSION", "2023-06-01"),
            },
        )
    if provider == "gemini":
        return (
            os.getenv(
                "MINDWEAVE_GEMINI_VALIDATION_URL",
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
            ),
            {},
        )
    if provider == "groq":
        return (
            os.getenv("MINDWEAVE_GROQ_VALIDATION_URL", "https://api.groq.com/openai/v1/models"),
            {"Authorization": f"Bearer {key}"},
        )
    if provider == "openrouter":
        return (
            os.getenv("MINDWEAVE_OPENROUTER_VALIDATION_URL", "https://openrouter.ai/api/v1/models"),
            {"Authorization": f"Bearer {key}"},
        )
    if provider == "xai":
        return (
            os.getenv("MINDWEAVE_XAI_VALIDATION_URL", "https://api.x.ai/v1/models"),
            {"Authorization": f"Bearer {key}"},
        )

    return (
        os.getenv("MINDWEAVE_OPENAI_VALIDATION_URL", "https://api.openai.com/v1/models"),
        {"Authorization": f"Bearer {key}"},
    )


def validate_mindweave_api_key(api_key: str) -> tuple[bool, str | None]:
    """Validate the user-supplied LLM API key across multiple providers."""
    key = api_key.strip()
    if not key:
        return False, "API key is required"

    if len(key) < 20:
        return False, "API key appears too short"

    provider = _infer_llm_provider(key)
    validation_url, headers = _build_validation_request(provider, key)
    timeout_seconds = float(os.getenv("MINDWEAVE_VALIDATION_TIMEOUT", "8"))

    try:
        response = requests.get(
            validation_url,
            headers=headers,
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        logger.warning("Mindweave key validation request failed for provider=%s: %s", provider, error)
        return False, "Unable to reach key validation service"

    if response.status_code == HTTPStatus.OK:
        return True, None

    if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
        return False, "API key rejected by provider"

    logger.warning(
        "Unexpected key validation response for provider=%s (%s): %s",
        provider,
        response.status_code,
        response.text[:200],
    )
    return False, f"Validation failed with status {response.status_code}"

class RequestHandler(SimpleHTTPRequestHandler):
    """Static file server + game launcher + unified AI APIs."""

    def __init__(self, *args, **kwargs):
        self.session_id = ""
        self._set_cookie_header: str | None = None
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _ensure_session(self) -> None:
        header = self.headers.get("Cookie", "")
        cookie = SimpleCookie()
        cookie.load(header)

        existing = cookie.get(SESSION_COOKIE)
        if existing and existing.value:
            self.session_id = existing.value
            return

        self.session_id = uuid.uuid4().hex
        self._set_cookie_header = (
            f"{SESSION_COOKIE}={self.session_id}; Path=/; HttpOnly; SameSite=Lax"
        )

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self._set_cookie_header:
            self.send_header("Set-Cookie", self._set_cookie_header)
            self._set_cookie_header = None
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._ensure_session()

        if self.path == "/api/games":
            self._send_json({"games": runtime.list_games()})
            return

        if self.path == "/api/selected-game":
            self._send_json(
                {
                    "session_id": self.session_id,
                    "selected_game": runtime.get_selected_game(self.session_id),
                }
            )
            return

        if self.path == "/api/ai/health":
            self._send_json(runtime.ai_health(self.session_id))
            return

        if self.path == "/":
            self.path = "/games/index.html"

        super().do_GET()

    def do_POST(self) -> None:
        self._ensure_session()

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON payload"}, HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/validate-mindweave-key":
            key = payload.get("api_key") if isinstance(payload, dict) else None
            if not isinstance(key, str):
                self._send_json({"valid": False, "error": "api_key must be a string"}, HTTPStatus.BAD_REQUEST)
                return

            is_valid, reason = validate_mindweave_api_key(key)
            if is_valid:
                self._send_json({"valid": True, "message": "API key verified"}, HTTPStatus.OK)
            else:
                self._send_json({"valid": False, "error": reason or "API key is invalid"}, HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/select-game":
            try:
                selected_game = payload.get("game")
                result = runtime.select_game(self.session_id, selected_game)
                self._send_json(result)
            except KeyError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as error:
                self._send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            except Exception as error:
                logger.exception("Failed during game selection")
                self._send_json(
                    {"error": f"Unable to initialize game: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if self.path == "/api/ai/move":
            try:
                referer = self.headers.get("Referer")
                result = runtime.ai_move(self.session_id, payload, referer=referer)
                self._send_json(result)
            except LookupError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except NotImplementedError as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_IMPLEMENTED)
            except Exception as error:
                logger.exception("AI move endpoint failed")
                self._send_json({"error": f"Move request failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path == "/api/ai/learn":
            try:
                result = runtime.ai_learn(self.session_id, payload)
                status = HTTPStatus.OK if result.get("status") != "error" else HTTPStatus.INTERNAL_SERVER_ERROR
                self._send_json(result, status)
            except LookupError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                logger.exception("AI learn endpoint failed")
                self._send_json({"error": f"Learn request failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path == "/api/ai/chat":
            try:
                if isinstance(payload, dict):
                    api_key = self.headers.get("X-Mindweave-API-Key", "").strip()
                    if api_key:
                        payload["api_key"] = api_key
                result = runtime.ai_chat(self.session_id, payload)
                status = HTTPStatus.OK if "error" not in result else HTTPStatus.BAD_REQUEST
                self._send_json(result, status)
            except LookupError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except NotImplementedError as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_IMPLEMENTED)
            except Exception as error:
                logger.exception("AI chat endpoint failed")
                self._send_json({"error": f"Chat request failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), RequestHandler)
    logger.info("R-Games launcher running at http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="R-Games launcher backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
