"""Thin Flask bridge for the Chronos SLAI AI runtime.

Heavy strategic logic belongs in ``games.ai_chronos``. This bridge keeps one
persistent ChronosAI instance alive, validates request payloads, and returns
stable JSON responses for the Chronos frontend/proxy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..ai_chronos import initialize_ai  # noqa: E402

app = Flask(__name__)
_ai = initialize_ai()


def _json_payload() -> tuple[dict[str, Any], str | None]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}, None
    if not isinstance(payload, dict):
        return {}, "JSON body must be an object"
    return payload, None


def _last_trace_summary() -> dict[str, Any] | None:
    trace = getattr(_ai, "last_trace", None)
    if not isinstance(trace, dict):
        return None
    return {
        "confidence": trace.get("confidence"),
        "fallback": trace.get("fallback"),
        "fallback_reason": trace.get("fallback_reason"),
        "strategic_objective": trace.get("strategic_objective"),
        "candidate_actions_count": trace.get("candidate_actions_count"),
        "safety_validation_result": trace.get("safety_validation_result"),
    }


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify({"status": "ok", "ai": _ai.health()}), 200


@app.post("/move")
def move() -> tuple[Any, int]:
    payload, error = _json_payload()
    if error:
        return jsonify({"move": None, "error": error, "fallback": True}), 400

    try:
        result = _ai.get_move(payload)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Chronos AI move request failed")
        return jsonify({"move": None, "error": f"AI move request failed: {exc}", "fallback": True}), 500

    if isinstance(result, dict) and "choice" in result and set(result.keys()) == {"choice"}:
        return jsonify({"choice": result["choice"], "trace": _last_trace_summary()}), 200

    # Debug/rich response requested by client.
    if isinstance(result, dict) and {"move", "confidence", "agent_trace", "fallback"}.issubset(result.keys()):
        return jsonify(result), 200

    if isinstance(result, dict):
        response: dict[str, Any] = {"move": result}
        if payload.get("include_trace_summary") or payload.get("debug"):
            response["trace"] = _last_trace_summary()
        return jsonify(response), 200

    return jsonify({"move": None, "fallback": True, "fallback_reason": "no legal move returned", "trace": _last_trace_summary()}), 200


@app.post("/learn")
def learn() -> tuple[Any, int]:
    payload, error = _json_payload()
    if error:
        return jsonify({"ok": False, "error": error}), 400

    try:
        success = _ai.learn_from_game(payload)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Chronos learning request failed")
        return jsonify({"ok": False, "error": f"Learning request failed: {exc}"}), 500

    status = 200 if success else 400
    return jsonify({"ok": bool(success), "stats": _ai.health().get("stats", {})}), status


@app.get("/training-status")
def training_status() -> tuple[Any, int]:
    status_fn = getattr(_ai, "training_status", None)
    if callable(status_fn):
        return jsonify(status_fn()), 200
    return jsonify(_ai.health().get("training_status", {"available": False})), 200


if __name__ == "__main__":
    port = int(os.getenv("PYTHON_PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=False)
