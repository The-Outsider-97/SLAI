import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logger = logging.getLogger('SafeAI.ModelTrainer')

class ModelTrainer:
    def train_model(self, data):
        logger.info("Training interpretable logistic regression model...")
        X, y = data
        try:
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            logger.info("Model trained successfully.")
            return model
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        logger.info("Evaluating model performance...")
        try:
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            logger.info(f"Model accuracy: {acc:.4f}")
            return acc
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
