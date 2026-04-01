from __future__ import annotations

import importlib
import json
import sys

from typing import Any, Dict


def _json_default(value: Any) -> str:
    return str(value)


def _build_agent(module_path: str, class_name: str):
    module = importlib.import_module(module_path)
    agent_cls = getattr(module, class_name)

    constructor_kwargs: Dict[str, Any] = {}
    try:
        from src.agents.agent_factory import AgentFactory
        from src.agents.collaborative.shared_memory import SharedMemory

        constructor_kwargs["shared_memory"] = SharedMemory()
        constructor_kwargs["agent_factory"] = AgentFactory()
    except Exception:
        # Keep worker resilient for agents that do not need these dependencies.
        pass

    return agent_cls(**constructor_kwargs)


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    module_path = payload["module_path"]
    class_name = payload["class_name"]
    method_name = payload["method"]
    args = payload.get("args", [])
    kwargs = payload.get("kwargs", {})

    try:
        agent = _build_agent(module_path, class_name)
        method = getattr(agent, method_name)
        result = method(*args, **kwargs)
        sys.stdout.write(json.dumps({"status": "ok", "result": result}, default=_json_default))
        return 0
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
