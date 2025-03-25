import os
import sys
import subprocess
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load default model at startup or on demand
def load_model(model_path="models/random_forest_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get("features")
        if not input_data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reload', methods=['POST'])
def reload():
    global model
    model_path = request.json.get("model_path") or "models/random_forest_model.pkl"
    try:
        model = load_model(model_path)
        return jsonify({"status": "Model reloaded successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
