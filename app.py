import os
import sys
import time
import glob
import logging
from flask import Flask, render_template, jsonify
from logger import get_log_queue

log_queue = queue.Queue()

class QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.queue = q

    def emit(self, record):
        msg = self.format(record)
        self.queue.put(msg)

handler = QueueLogHandler(log_queue)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

app = Flask(
    __name__,
    template_folder='frontend/templates',
    static_folder='frontend/styles'
)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/metrics')
def stream_metrics():
    dummy_metrics = {
        "accuracy": 0.542,
        "risk_score": 0.27,
        "reward": 129.6
    }
    return jsonify(dummy_metrics)

@app.route('/logs')
def stream_logs():
    log_queue = get_log_queue()
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return jsonify(logs)

@app.route('/launch', methods=['POST'])
def launch():
    data = request.get_json()
    filename = data.get('file')

    if not filename or not filename.endswith('.py'):
        return jsonify({"error": "Invalid file selected."}), 400

    try:
        # Launch the script asynchronously
        subprocess.Popen(['python', filename])
        return jsonify({"status": "launched", "file": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sessions')
def list_sessions():
    session_files = glob.glob("logs/hparam_trials*.jsonl")
    return jsonify(session_files)

@app.route('/session/<session_name>')
def get_session(session_name):
    path = os.path.join("logs", session_name)
    if not os.path.exists(path):
        return jsonify({"error": "Session not found"}), 404
    with open(path, "r") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return jsonify(lines)

if __name__ == '__main__':
    app.run(debug=True)
