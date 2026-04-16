"""Chronos AI HTTP bridge.

This service keeps a persistent `ChronosAI` instance alive and exposes
`/move` and `/learn` endpoints used by the Chronos web server proxy.
It also exposes training-progress status so the web UI can show live
training updates without relying on terminal output.
"""

from __future__ import annotations

import json
import os
import sys
import time

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from games.ai_chronos import initialize_ai


app = Flask(__name__)
_ai = initialize_ai()
_logs_dir = PROJECT_ROOT / "logs"
_training_checkpoint_path = _logs_dir / "chronos_training_checkpoint.json"
_training_summary_path = _logs_dir / "chronos_training_summary.json"


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify({"status": "ok", "ai": _ai.health()}), 200


@app.post("/move")
def move() -> tuple[Any, int]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}

    result = _ai.get_move(payload)
    if isinstance(result, dict):
        if "choice" in result:
            return jsonify({"choice": result["choice"]}), 200
        return jsonify({"move": result}), 200
    return jsonify({"move": None}), 200


@app.post("/learn")
def learn() -> tuple[Any, int]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}

    success = _ai.learn_from_game(payload)
    return jsonify({"ok": bool(success)}), 200


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


@app.get("/training-status")
def training_status() -> tuple[Any, int]:
    checkpoint = _load_json(_training_checkpoint_path)
    summary = _load_json(_training_summary_path)

    completed_episodes = int(checkpoint.get("completed_episodes", summary.get("completed_episodes", 0)) or 0)
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint.get("meta"), dict) else {}
    total_episodes = int(
        meta.get(
            "episodes",
            summary.get("episodes", 0),
        )
        or 0
    )
    progress = round((completed_episodes / total_episodes) * 100.0, 2) if total_episodes > 0 else 0.0

    checkpoint_mtime = _training_checkpoint_path.stat().st_mtime if _training_checkpoint_path.exists() else 0.0
    seconds_since_update = max(0.0, time.time() - checkpoint_mtime) if checkpoint_mtime else None
    active = seconds_since_update is not None and seconds_since_update <= 15.0 and completed_episodes < total_episodes

    status = {
        "available": bool(checkpoint or summary),
        "active": bool(active),
        "completed_episodes": completed_episodes,
        "total_episodes": total_episodes,
        "progress_percent": progress,
        "depth": int(meta.get("depth", summary.get("depth", 0)) or 0),
        "counter_playouts": int(meta.get("playouts", summary.get("counter_playouts", 0)) or 0),
        "explored_states": int(summary.get("explored_states", 0) or 0),
        "counter_evals": int(summary.get("counter_evals", 0) or 0),
        "wins": summary.get("wins", {}),
        "losses": summary.get("losses", {}),
        "draw_like": summary.get("draw_like", {}),
        "seconds_since_update": None if seconds_since_update is None else round(seconds_since_update, 2),
        "checkpoint_path": str(_training_checkpoint_path),
        "summary_path": str(_training_summary_path),
    }
    return jsonify(status), 200


if __name__ == "__main__":
    port = int(os.getenv("PYTHON_PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=False)
