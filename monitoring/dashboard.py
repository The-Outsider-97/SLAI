from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import logging
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
socketio = SocketIO(app)

# Stores the last X results
MAX_HISTORY = 50
rsi_history = []


@app.route('/')
def index():
    return render_template('dashboard.html')


def push_rsi_update(iteration, reward, risk_level, details=None):
    """
    Push a real-time update to the dashboard.
    """
    global rsi_history

    update = {
        "iteration": iteration,
        "reward": reward,
        "risk_level": risk_level,
        "details": details or {}
    }

    rsi_history.append(update)
    if len(rsi_history) > MAX_HISTORY:
        rsi_history.pop(0)

    logger.info(f"Pushing update: Iter {iteration} | Reward {reward} | Risk {risk_level}")
    socketio.emit('rsi_update', update)


def run_dashboard():
    """
    Starts the Flask dashboard server.
    """
    logger.info("Starting RSI Monitoring Dashboard on http://localhost:5000")
    socketio.run(app, debug=False, port=5000)
