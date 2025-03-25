import logging
import joblib
import os
import sys
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from modules.deployment.model_registry import register_model

logger = logging.getLogger('SafeAI.ModelTrainer')
logger.setLevel(logging.INFO)

class ModelTrainer:
    def __init__(self, shared_memory=None, output_dir="models/"):
        self.shared_memory = shared_memory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train_model(self, data, model_type="logistic", params=None, test_size=0.2, random_state=42):
        """
        Train a classifier using provided features and labels.

        Args:
            data: tuple (X, y)
            model_type: "logistic" or "random_forest"
            params: dictionary of model hyperparameters
            test_size: split ratio for validation
        """
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        logger.info(f"Training model [{model_type}]...")

        try:
            if model_type == "logistic":
                model = LogisticRegression(max_iter=200, **(params or {}))
            elif model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, **(params or {}))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            model.fit(X_train, y_train)
            logger.info("Model training completed.")

            metrics = self.evaluate_model(model, X_test, y_test)

            # Persist model
            model_path = os.path.join(self.output_dir, f"{model_type}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved to: {model_path}")

            # Register model (moved here from global scope)
            try:
                register_model(
                    model_name=f"{model_type}_v1",
                    path=model_path,
                    metadata={
                        "accuracy": metrics["accuracy"],
                        "type": model_type
                    }
                )
            except Exception as e:
                logger.warning(f"Model registration failed: {e}")

            # Share results
            if self.shared_memory:
                self.shared_memory.set("model_metrics", metrics)
                self.shared_memory.set("trained_model_path", model_path)

            return model, metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a trained model with multiple metrics.
        """
        logger.info("Evaluating model...")
        try:
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = {
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "f1_score": round(f1_score(y_test, preds), 4),
                "roc_auc": round(roc_auc_score(y_test, proba), 4) if proba is not None else "N/A",
                "confusion_matrix": confusion_matrix(y_test, preds).tolist()
            }

            logger.info(f"Model Evaluation Metrics:\n{metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def load_model(self, model_path):
        """
        Load a model from disk.
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
