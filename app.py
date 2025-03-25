import os
import sys
import time
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

# Configure global logger
logger = logging.getLogger("SLAI")
logger.setLevel(logging.INFO)

# Attach queue handler
queue_handler = QueueLogHandler(log_queue)
queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(queue_handler)

# Optional: also log to file or console
file_handler = logging.FileHandler("logs/slai.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Utility access
get_logger = lambda: logger
get_log_queue = lambda: log_queue

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

if __name__ == '__main__':
    app.run(debug=True)
