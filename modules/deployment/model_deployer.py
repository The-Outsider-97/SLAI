import os
import sys
import torch
import logging
import numpy as np
import joblib

logger = logging.getLogger("SafeAI.ModelDeployer")
logger.setLevel(logging.INFO)

class ModelDeployer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load(self):
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model is not loaded. Call `.load()` first.")

        try:
            input_array = np.array(input_data).reshape(1, -1)
            prediction = self.model.predict(input_array)[0]
            return {"prediction": int(prediction)}
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
