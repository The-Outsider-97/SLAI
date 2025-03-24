import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('SafeAI.DataHandler')

class DataHandler:
    def load_data(self, path):
        logger.info(f"Loading dataset from {path}")
        try:
            data = pd.read_csv(path)
            logger.info("Data loaded successfully.")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def check_data_fairness(self, data):
        logger.info("Checking data for fairness and bias...")
        # Placeholder for actual bias checking logic
        sensitive_columns = ['gender', 'race']
        for col in sensitive_columns:
            if col in data.columns:
                logger.info(f"Distribution for {col}:\n{data[col].value_counts(normalize=True)}")
            else:
                logger.warning(f"Column {col} not found in data.")

    def preprocess_data(self, data):
        logger.info("Preprocessing dataset...")
        try:
            features = data.drop(columns=['label'])
            labels = data['label']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            logger.info("Data preprocessing complete.")
            return pd.DataFrame(scaled_features), labels
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
