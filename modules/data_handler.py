
import re
import csv
import math
import numpy as np
import pandas as pd
import os, sys, psutil

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from collections import defaultdict
from typing import List, Dict, Tuple

from src.agents.collaborative.shared_memory import SharedMemory
from logs.logger import get_logger

logger = get_logger('SafeAI.DataHandler')

class DataHandler:
    def __init__(self, chunk_size=10000, max_rows=None, config=None):
        self.shared_memory = SharedMemory(config)
        self.chunk_size = chunk_size
        self.max_rows = max_rows
        self.scaler = None

        self.imputer = None
        self.encoder = None
        self.fairness_metrics = {}
        self._schema_rules = {}
        self.resource_history = {}

    def _log_memory_usage(self, label=""):
        """Extended memory, CPU, and disk I/O tracking"""
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 2)
        cpu = psutil.cpu_percent()
        disk = psutil.disk_io_counters()

        logger.info(f"[Resource - {label}] RAM: {mem:.2f} MB | CPU: {cpu:.1f}% | Disk Read: {disk.read_bytes} B | Write: {disk.write_bytes} B")

        if self.shared_memory:
            self.shared_memory.set(f"memory_usage_{label}", {
                "ram_mb": round(mem, 2),
                "cpu_percent": cpu,
                "disk_read": disk.read_bytes,
                "disk_write": disk.write_bytes
            })

        self.resource_history[label] = {
            'timestamp': pd.Timestamp.now(),
            'cpu': cpu,
            'disk_read': disk.read_bytes,
            'disk_write': disk.write_bytes,
            'memory_mb': round(mem, 2)
        }

    def _get_numeric_columns(self, data):
        """Return only numeric columns (float and int)"""
        return data.select_dtypes(include=['float64', 'int64']).columns

    def load_data(self, path, shuffle_data=True):
        """Load CSV file in chunks and return concatenated DataFrame"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Loading dataset from: {path}")
        chunks = []
        total_rows = 0

        try:
            for chunk in pd.read_csv(path, chunksize=self.chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                self._log_memory_usage(f"chunk-{total_rows}")
                if self.max_rows and total_rows >= self.max_rows:
                    break
        except Exception as e:
            raise RuntimeError(f"Failed during chunk loading: {e}")

        data = pd.concat(chunks, ignore_index=True)
        del chunks
        self._log_memory_usage("after_concat")

        if shuffle_data:
            data = shuffle(data).reset_index(drop=True)

        return data

    def preprocess_data(self, data, scale=True, label_column="label"):
        """Preprocess dataset and scale features if needed"""
        logger.info("Preprocessing dataset...")
        self._log_memory_usage("preprocessing_start")

        try:
            if label_column not in data.columns:
                raise ValueError(f"Label column '{label_column}' not found in dataset")

            numeric_cols = self._get_numeric_columns(data)
            if scale:
                scaled_values = self.scaler.fit_transform(data[numeric_cols])
                data[numeric_cols] = scaled_values
                self._log_memory_usage("after_scaling")

            features = data.drop(columns=[label_column])
            labels = data[label_column]

            numeric_features = features.select_dtypes(include=[np.number])
            scaled = self.scaler.fit_transform(numeric_features)

            logger.info(f"Preprocessed data shape: {scaled.shape}")
            self._log_memory_usage("after_preprocessing")
            return pd.DataFrame(scaled, columns=numeric_features.columns), labels

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def _check_missing_values(self, data):
        missing = data.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            logger.warning(f"Missing values detected:\n{missing}")
            if self.shared_memory:
                self.shared_memory.set("data_warnings", {"missing_values": missing.to_dict()})

    # ========== Preprocessing ==========
    def set_preprocessing_strategy(self, strategy: Dict):
        """Configurable preprocessing pipeline
        Args:
            strategy: {
                'scaling': 'standard|minmax|robust',
                'imputation': 'mean|median|knn(k=3)',
                'encoding': 'onehot|label'
            }
        """
        # Custom scaler implementations
        if 'scaling' in strategy:
            if strategy['scaling'] == 'standard':
                self.scaler = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            elif strategy['scaling'] == 'minmax':
                self.scaler = lambda x: (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
        
        # Imputation methods
        if 'imputation' in strategy:
            if strategy['imputation'].startswith('knn'):
                match = re.search(r'k=(\d+)', strategy['imputation'])
                if match:
                    k = int(match.group(1))
                    self.imputer = self._knn_imputer(k=k)
                else:
                    raise ValueError("Invalid format for KNN imputation. Use 'knn(k=3)'")
                
    def _knn_imputer(self, k=3):
        """Basic KNN imputation without external dependencies"""
        def impute(chunk):
            # Simplified implementation for numeric columns only
            distances = np.sqrt(np.sum((chunk[:, None] - chunk) ** 2, axis=2))
            for i in range(chunk.shape[0]):
                missing = np.isnan(chunk[i])
                if np.any(missing):
                    neighbors = np.argsort(distances[i])[1:k+1]
                    chunk[i, missing] = np.nanmean(chunk[neighbors, missing], axis=0)
            return chunk
        return impute

    def check_data_fairness(self, data, sensitive_columns=None):
        """quantitative fairness metrics"""
        logger.info("Checking data for fairness and bias...")
        self._log_memory_usage("fairness_check_start")
        super().check_data_fairness(data, sensitive_columns)

        if sensitive_columns:
            for col in sensitive_columns:
                if col in data.columns:
                    self.fairness_metrics.update({
                        f'{col}_disparate_impact': self._disparate_impact_ratio(data, col),
                        f'{col}_statistical_parity': self._statistical_parity_difference(data, col)
                    })

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

    def _disparate_impact_ratio(self, data, sensitive_col):
        """Compute (P(positive|minority)/P(positive|majority)"""
        positive_rate = data.groupby(sensitive_col)['label'].mean()
        return positive_rate.min() / positive_rate.max()

    def _statistical_parity_difference(self, data, sensitive_col):
        """Max - min positive rates across groups"""
        return data.groupby(sensitive_col)['label'].mean().ptp()

    # ========== Schema Validation ==========
    def add_schema_rule(self, field: str, checks: Dict):
        """Add validation rules for specific fields
        Example:
            add_schema_rule('age', {'dtype': 'int', 'min': 0, 'max': 120})
        """
        self._schema_rules[field] = checks

    def validate_schema(self, data, required_columns):
        """Enhanced with type and range checks"""
        base_result = super().validate_schema(data, required_columns)
        
        # New checks
        for col, rules in self._schema_rules.items():
            if col in data.columns:
                # Type checking
                if 'dtype' in rules:
                    if not np.issubdtype(data[col].dtype, np.dtype(rules['dtype'])):
                        logger.warning(f"Type mismatch for {col}")
                        
                # Range checking
                if 'min' in rules:
                    if data[col].min() < rules['min']:
                        logger.warning(f"Values below min in {col}")
                # Similar for max
        return base_result and len(self._schema_rules) == 0

    # ========== Resource Optimization ==========
    def adaptive_chunking(self, initial_chunk=1000):
        """Dynamically adjust chunk size based on memory"""
        self.chunk_size = initial_chunk
        while True:
            try:
                with pd.read_csv(self.path, chunksize=self.chunk_size) as reader:
                    next(reader)  # Test read
                    break
            except MemoryError:
                self.chunk_size = int(self.chunk_size * 0.8)
                logger.info(f"Reduced chunk size to {self.chunk_size}")

    # ========== Data Augmentation ==========
    def balance_classes(self, data, label_col='label', method='smote'):
        """Basic class balancing implementation"""
        if method == 'smote':
            return self._simple_smote(data, label_col)
        # Other methods...

    def _simple_smote(self, data, label_col, k=5):
        """Synthetic Minority Over-sampling Technique (basic)"""
        minority = data[data[label_col] == 1]
        synthetic = []
        for _ in range(len(data) - 2*len(minority)):
            i, j = np.random.choice(len(minority), 2, replace=False)
            synthetic_sample = minority.iloc[i] + (minority.iloc[j] - minority.iloc[i]) * np.random.rand()
            synthetic.append(synthetic_sample)
        return pd.concat([data] + synthetic)

if __name__ == '__main__':
    handler = DataHandler()
    handler.add_schema_rule('age', {'dtype': 'int', 'min': 18})
    handler.set_preprocessing_strategy({
        'scaling': 'minmax',
        'imputation': 'knn(k=3)'
    })

    data = handler.load_data("data/users.csv")
    data = handler.preprocess_data(data, label_column="target")
    fairness_report = handler.check_data_fairness(data)
