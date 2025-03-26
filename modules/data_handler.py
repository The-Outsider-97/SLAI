import pandas as pd
import numpy as np
import logging
import os, sys
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

logger = logging.getLogger('SafeAI.DataHandler')
logger.setLevel(logging.INFO)

class DataHandler:
    def __init__(self, shared_memory=None):
        self.shared_memory = shared_memory
        self.scaler = StandardScaler()

    def load_data(self, path, shuffle_data=True):
        logger.info(f"Loading dataset from {path}")
        try:
            data = pd.read_csv(path)
            if shuffle_data:
                data = shuffle(data)
            logger.info(f"Data loaded. Shape: {data.shape}")
            self._check_missing_values(data)
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _check_missing_values(self, data):
        missing = data.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            logger.warning(f"Missing values detected:\n{missing}")
            if self.shared_memory:
                self.shared_memory.set("data_warnings", {"missing_values": missing.to_dict()})

    def check_data_fairness(self, data, sensitive_columns=None):
        logger.info("Checking data for fairness and bias...")

        if sensitive_columns is None:
            sensitive_columns = ['gender', 'race', 'ethnicity', 'age']

        fairness_report = {}
        for col in sensitive_columns:
            if col in data.columns:
                distribution = data[col].value_counts(normalize=True)
                logger.info(f"{col} distribution:\n{distribution}")
                fairness_report[col] = distribution.to_dict()
            else:
                logger.warning(f"Column {col} not found in data.")
                fairness_report[col] = "Missing"

        if self.shared_memory:
            self.shared_memory.set("fairness_check", fairness_report)

        return fairness_report

    def preprocess_data(self, data, label_column="label"):
        logger.info("Preprocessing dataset...")

        try:
            if label_column not in data.columns:
                raise ValueError(f"Label column '{label_column}' not found in dataset")

            features = data.drop(columns=[label_column])
            labels = data[label_column]

            numeric_features = features.select_dtypes(include=[np.number])
            scaled = self.scaler.fit_transform(numeric_features)

            logger.info(f"Preprocessed data shape: {scaled.shape}")
            return pd.DataFrame(scaled, columns=numeric_features.columns), labels
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def validate_schema(self, data, required_columns):
        logger.info("Validating dataset schema...")

        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            if self.shared_memory:
                self.shared_memory.set("schema_warnings", {"missing_columns": missing})
            return False
        return True
