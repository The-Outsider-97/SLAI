"""
monitoring/dashboard.py
────────────────────────
Real-time Flask/SocketIO dashboard for SLAI monitoring.

Key improvements over v1
─────────────────────────
  • History loaded on connect – new clients see the last MAX_HISTORY points
  • Thread-safe history access
  • Accepts full MetricSnapshot dicts (not just RSI reward/risk scalars)
  • push_metrics_update() helper bridges the MetricsCollector loop
  • Structured logging via StructuredLogger
  • Configurable port / host via MonitoringConfig (or env vars)
"""

from __future__ import annotations

from pathlib import Path
import threading

from typing import Any
from flask import Flask, render_template, Response, jsonify, send_from_directory
from flask_socketio import SocketIO, emit # type: ignore

from .config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Dashboard")
printer = PrettyPrinter()


# ──────────────────────────────────────────────
# Module state
# ──────────────────────────────────────────────
_FLASK_AVAILABLE = True
_lock = threading.Lock()
_history: list[dict[str, Any]] = []
MAX_HISTORY = 50
_rsi_history: list[dict[str, Any]] = []

# Read configuration
config = get_config_section("dashboard")
MAX_HISTORY = config.get("max_history", 50)
HOST = config.get("host", "0.0.0.0")
PORT = config.get("port", 5000)

app: Any = None
socketio: Any = None

# ──────────────────────────────────────────────
# Create Flask app only if dependencies are present
# ──────────────────────────────────────────────
if _FLASK_AVAILABLE:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "slai-monitor-secret"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    @app.route("/")
    def index():
        try:
            return render_template("dashboard.html")
        except Exception as e:
            # Template missing – return a helpful error page
            return f"<html><body><h1>Dashboard template missing</h1><p>Please create monitoring/templates/dashboard.html</p><pre>{e}</pre></body></html>", 500

    @app.route("/metrics/prometheus")
    def prometheus_metrics():
        with _lock:
            history = list(_history)
        if not history:
            return Response("# No snapshots collected yet.\n", mimetype="text/plain")
        # Find the latest metrics update
        for item in reversed(history):
            if item.get("type") == "metrics":
                prom_text = item.get("prometheus", "# Prometheus data unavailable.\n")
                return Response(prom_text, mimetype="text/plain; version=0.0.4")
        return Response("# No metrics snapshot available.\n", mimetype="text/plain")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "history_size": len(_history)})

    @socketio.on("connect")
    def on_connect():
        with _lock:
            history_snapshot = list(_history)
        emit("history", history_snapshot)
        logger.debug("Dashboard client connected.", history_size=len(history_snapshot))

    @socketio.on("disconnect")
    def on_disconnect():
        logger.debug("Dashboard client disconnected.")

# ──────────────────────────────────────────────
# Public update helpers
# ──────────────────────────────────────────────
def push_rsi_update(
    iteration: int,
    reward: float,
    risk_level: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Push an RSI training update."""
    update = {
        "type": "rsi",
        "iteration": iteration,
        "reward": reward,
        "risk_level": risk_level,
        "details": details or {},
    }
    _append_and_emit(update)
    logger.info(f"RSI update pushed. iteration={iteration}, reward={reward}, risk_level={risk_level}")


def push_metrics_update(snapshot_dict: dict[str, Any]) -> None:
    """Push a MetricSnapshot dict to the dashboard."""
    update = {"type": "metrics", **snapshot_dict}
    _append_and_emit(update)


def push_alert(subject: str, severity: str, message: str) -> None:
    """Push an alert notification."""
    update = {"type": "alert", "subject": subject, "severity": severity, "message": message}
    _append_and_emit(update)


def _append_and_emit(update: dict[str, Any]) -> None:
    with _lock:
        _history.append(update)
        if len(_history) > MAX_HISTORY:
            _history.pop(0)
    if _FLASK_AVAILABLE and socketio is not None:
        socketio.emit("rsi_update", update)


# ──────────────────────────────────────────────
# Server launchers
# ──────────────────────────────────────────────
def run_dashboard(host: str = None, port: int = None) -> None:
    """Start the dashboard server (blocking)."""
    if not _FLASK_AVAILABLE:
        logger.error("Cannot start dashboard: Flask or flask-socketio not installed.")
        return
    host = host or HOST
    port = port or PORT
    logger.info(f"Starting SLAI Monitoring Dashboard. host={host}, port={port}")
    try:
        socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error("Dashboard failed to start.", error=str(e))


def run_dashboard_thread(host: str = None, port: int = None) -> threading.Thread | None:
    """Launch the dashboard in a background daemon thread."""
    if not _FLASK_AVAILABLE:
        logger.error("Cannot start dashboard thread: Flask/flask-socketio not installed.")
        return None
    t = threading.Thread(
        target=run_dashboard,
        kwargs={"host": host, "port": port},
        daemon=True,
        name="slai-dashboard",
    )
    t.start()
    return t

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@app.route("/component/assets/<path:filename>")
def component_assets(filename):
    return send_from_directory(PROJECT_ROOT / "component" / "assets", filename)

if __name__ == "__main__":
    # When run directly, start the dashboard in the main thread
    if _FLASK_AVAILABLE:
        run_dashboard()
    else:
        print("Flask or flask-socketio not installed. Install with: pip install flask flask-socketio")