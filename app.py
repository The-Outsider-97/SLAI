from flask import Flask, render_template
from flask import jsonify
import time

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
    # Replace this with reading from a file or a logging queue
    dummy_logs = [
        "INFO  Starting Evolutionary Generation 7",
        "INFO  Gen 7 accuracy: 52.6%",
        "INFO  Training completed."
    ]
    return jsonify(dummy_logs)
    
app = Flask(
    __name__,
    template_folder='frontend/templates',
    static_folder='frontend/styles')

@app.route('/')
def index():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
