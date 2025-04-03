import os
import json
import yaml
import csv
import pickle
import torch
import pandas as pd
import pyarrow.parquet as pq
from typing import Any, Dict, List, Optional, Callable, Union
import logging
from functools import lru_cache
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

# ============================================
# Logger Configuration
# ============================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ============================================
# Supported Formats
# ============================================
SUPPORTED_FORMATS = ["json", "yaml", "csv", "parquet", "pickle"]

# ============================================
# Dummy Data Generator
# ============================================
def generate_dummy_data(batch_size: int = 32, num_batches: int = 10, input_size: int = 10, output_size: int = 2) -> TorchDataLoader:
    """
    Generates dummy data loaders for training and validation.

    Args:
        batch_size (int): Number of samples per batch.
        num_batches (int): Number of batches.
        input_size (int): Number of input features.
        output_size (int): Number of output classes (as class indices).

    Returns:
        TorchDataLoader: PyTorch DataLoader containing the dummy dataset.
    """
    logger.debug(f"Generating dummy data with batch_size={batch_size}, num_batches={num_batches}, input_size={input_size}, output_size={output_size}")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
    if not isinstance(num_batches, int) or num_batches <= 0:
        raise ValueError(f"num_batches must be a positive integer, got {num_batches}")
    if input_size <= 0 or output_size <= 0:
        raise ValueError("input_size and output_size must be positive integers")

    total_samples = batch_size * num_batches

    X = torch.randn(total_samples, input_size)
    y = torch.randint(0, output_size, (total_samples,))

    dataset = TensorDataset(X, y)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Dummy dataset created with {total_samples} samples.")
    return loader

# ============================================
# Flexible Data Loader Class
# ============================================
class FlexibleDataLoader:
    """
    Flexible DataLoader that supports JSON, YAML, CSV, Parquet, Pickle, and Pandas DataFrames.
    Provides optional schema validation, preprocessing, caching, and batch loading.
    """

    def __init__(self, validation_schema: Optional[Dict[str, type]] = None):
        self.validation_schema = validation_schema or {}
        logger.info(f"FlexibleDataLoader initialized with schema: {self.validation_schema}")

    def load(self, file_path: str, preprocess_fn: Optional[Callable[[Any], Any]] = None) -> Any:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1][1:].lower()
        logger.info(f"Loading file '{file_path}' with extension '{ext}'")

        if ext not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")

        load_method = getattr(self, f"_load_{ext}")
        data = load_method(file_path)

        if self.validation_schema:
            self._validate_schema(data)

        if preprocess_fn:
            logger.info("Applying preprocessing function to data...")
            data = preprocess_fn(data)

        logger.info(f"File '{file_path}' loaded successfully.")
        return data

    @staticmethod
    def _load_json(file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("JSON data loaded.")
        return data

    @staticmethod
    def _load_yaml(file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.debug("YAML data loaded.")
        return data

    @staticmethod
    def _load_csv(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        logger.debug(f"CSV data loaded with shape {df.shape}")
        return df

    @staticmethod
    def _load_parquet(file_path: str) -> pd.DataFrame:
        df = pd.read_parquet(file_path)
        logger.debug(f"Parquet data loaded with shape {df.shape}")
        return df

    @staticmethod
    def _load_pickle(file_path: str) -> Any:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.debug("Pickle data loaded.")
        return data

    def _validate_schema(self, data: Union[Dict, pd.DataFrame]):
        logger.info("Validating data against schema...")

        if isinstance(data, dict):
            for key, expected_type in self.validation_schema.items():
                if key not in data:
                    logger.error(f"Schema validation failed: missing key '{key}'")
                    raise ValueError(f"Missing key in data: {key}")
                if not isinstance(data[key], expected_type):
                    logger.error(f"Schema validation failed: key '{key}' expected {expected_type}, got {type(data[key])}")
                    raise ValueError(f"Invalid type for key '{key}': expected {expected_type}, got {type(data[key])}")
        elif isinstance(data, pd.DataFrame):
            for col in self.validation_schema.keys():
                if col not in data.columns:
                    logger.error(f"Schema validation failed: missing column '{col}'")
                    raise ValueError(f"Missing column in DataFrame: {col}")
        else:
            logger.error("Unsupported data type for schema validation.")
            raise TypeError("Schema validation not supported for this data type.")

        logger.info("Schema validation passed.")

    @lru_cache(maxsize=32)
    def cached_load(self, file_path: str, preprocess_fn: Optional[Callable[[Any], Any]] = None) -> Any:
        logger.info(f"Loading cached file '{file_path}'")
        return self.load(file_path, preprocess_fn)

    @staticmethod
    def batch_load(file_paths: List[str], preprocess_fn: Optional[Callable[[Any], Any]] = None) -> List[Any]:
        logger.info(f"Batch loading {len(file_paths)} files...")
        loader = FlexibleDataLoader()
        results = []

        for path in file_paths:
            try:
                data = loader.load(path, preprocess_fn)
                results.append(data)
                logger.info(f"Loaded '{path}' successfully.")
            except Exception as e:
                logger.error(f"Failed to load '{path}': {e}")

        logger.info("Batch loading completed.")
        return results

DataLoader = FlexibleDataLoader
