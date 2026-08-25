"""Thin Flask bridge for the Aether Shift SLAI runtime.

The shared R-Games backend remains the preferred integration path. This bridge
is provided for standalone Aether serving and mirrors the common `/health`,
`/move`, `/learn`, and `/training-status` contracts without duplicating game AI
logic from `games/ai_aether.py`.
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

from ..ai_aether import initialize_ai  # noqa: E402

app = Flask(__name__)
_ai = initialize_ai()


def _json_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


@app.get('/health')
def health() -> tuple[Any, int]:
    return jsonify({'status': 'ok', 'ai': _ai.health()}), 200


@app.post('/move')
def move() -> tuple[Any, int]:
    payload = _json_payload()
    response = _ai.get_move_response(payload)
    # Preserve raw move compatibility while exposing optional diagnostics.
    return jsonify(response), 200


@app.post('/learn')
def learn() -> tuple[Any, int]:
    payload = _json_payload()
    success = _ai.learn_from_game(payload)
    return jsonify({'ok': bool(success)}), 200


@app.get('/training-status')
def training_status() -> tuple[Any, int]:
    return jsonify(_ai.training_status()), 200


if __name__ == '__main__':
    port = int(os.getenv('PYTHON_PORT', '5001'))
    app.run(host='127.0.0.1', port=port, debug=False)
