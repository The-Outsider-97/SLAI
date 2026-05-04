from __future__ import annotations

import json
import subprocess
import sys, os

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.agents.factory.utils.config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Out Of Process Agent")
printer = PrettyPrinter


@dataclass
class OutOfProcessAgentProxy:
    """
    Runs agent calls in an isolated Python subprocess so native DLL failures
    (e.g. torch c10.dll on Windows) do not crash the main app process.
    """

    agent_type: str
    module_path: str
    class_name: str
    init_error: str

    implementation: str = "out_of_process_proxy"

    def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        payload: Dict[str, Any] = {
            "module_path": self.module_path,
            "class_name": self.class_name,
            "method": method,
            "args": args,
            "kwargs": kwargs,
        }
        try:
            child_env = os.environ.copy()
            # Ensure worker stdio uses UTF-8 to avoid Windows cp1252/charmap crashes
            # when dependencies/loggers emit non-ASCII symbols (e.g. ℹ).
            child_env.setdefault("PYTHONIOENCODING", "utf-8")
            child_env.setdefault("PYTHONUTF8", "1")
            completed = subprocess.run(
                [sys.executable, "-m", "src.agents.factory.remote_worker"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
                env=child_env,
            )
        except Exception as exc:
            logger.warning(
                "Out-of-process call failed for '%s.%s': %s",
                self.agent_type,
                method,
                exc,
            )
            return {"status": "degraded", "agent": self.agent_type, "error": str(exc)}

        if completed.returncode != 0:
            err = completed.stderr.strip() or completed.stdout.strip()
            logger.warning(
                "Out-of-process agent '%s' failed (%s): %s",
                self.agent_type,
                method,
                err,
            )
            return {"status": "degraded", "agent": self.agent_type, "error": err}

        raw = completed.stdout.strip()
        if not raw:
            return {"status": "ok", "agent": self.agent_type, "result": None}
        try:
            response = json.loads(raw)
            return response.get("result", response)
        except json.JSONDecodeError:
            return {"status": "degraded", "agent": self.agent_type, "error": raw}

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("execute", *args, **kwargs)

    def perform_task(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("perform_task", *args, **kwargs)

    def act(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("act", *args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("predict", *args, **kwargs)

    def get_action(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("get_action", *args, **kwargs)

    def failure_normalization(self, *args: Any, **kwargs: Any) -> Any:
        # This is frequently used by SignalSentry; keep best-effort semantics.
        return self._invoke("failure_normalization", *args, **kwargs)

    def __getattr__(self, item: str):
        def _call(*args: Any, **kwargs: Any):
            return self._invoke(item, *args, **kwargs)

        return _call
