import os
import json
import yaml
import csv
import pickle
import pandas as pd
import pyarrow.parquet as pq
from typing import Any, Dict, List, Optional, Callable, Union
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SUPPORTED_FORMATS = ["json", "yaml", "csv", "parquet", "pickle"]

class DataLoader:
    """
    Flexible DataLoader for various file types.
    
    Supports JSON, YAML, CSV, Parquet, Pickle, and Pandas DataFrames.
    """

    def __init__(self, validation_schema: Optional[Dict[str, type]] = None):
        """
        Initializes the DataLoader.
        
        Parameters:
        - validation_schema (dict): Optional schema for validation.
        """
        self.validation_schema = validation_schema or {}
        logger.info(f"DataLoader initialized with schema: {self.validation_schema}")

    def load(self, file_path: str, preprocess_fn: Optional[Callable[[Any], Any]] = None) -> Any:
        """
        Loads data from a file with optional preprocessing and schema validation.

        Parameters:
        - file_path (str): Path to the data file.
        - preprocess_fn (Callable): Function to apply to the data after loading.

        Returns:
        - Loaded and optionally preprocessed data.
        """

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1][1:].lower()

        logger.info(f"Loading file {file_path} with extension {ext}")

        if ext not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")

        load_method = getattr(self, f"_load_{ext}")
        data = load_method(file_path)

        # Schema validation
        if self.validation_schema:
            self._validate_schema(data)

        # Preprocess (if any)
        if preprocess_fn:
            logger.info("Applying preprocessing function to data...")
            data = preprocess_fn(data)

        logger.info(f"File {file_path} loaded successfully.")
        return data

    @staticmethod
    def _load_json(file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"JSON data loaded: {data}")
        return data

    @staticmethod
    def _load_yaml(file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.debug(f"YAML data loaded: {data}")
        return data

    @staticmethod
    def _load_csv(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        logger.debug(f"CSV data loaded: {df.shape} rows x {df.shape[1]} columns")
        return df

    @staticmethod
    def _load_parquet(file_path: str) -> pd.DataFrame:
        df = pd.read_parquet(file_path)
        logger.debug(f"Parquet data loaded: {df.shape} rows x {df.shape[1]} columns")
        return df

    @staticmethod
    def _load_pickle(file_path: str) -> Any:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.debug(f"Pickle data loaded.")
        return data

    def _validate_schema(self, data: Union[Dict, pd.DataFrame]):
        """
        Validates loaded data against the provided schema.

        Raises ValueError if validation fails.
        """
        logger.info("Validating data against schema...")

        if isinstance(data, dict):
            for key, expected_type in self.validation_schema.items():
                if key not in data:
                    logger.warning(f"Schema validation failed: missing key '{key}'")
                    raise ValueError(f"Missing key in data: {key}")
                if not isinstance(data[key], expected_type):
                    logger.warning(f"Schema validation failed: key '{key}' expected {expected_type}, got {type(data[key])}")
                    raise ValueError(f"Invalid type for key '{key}': expected {expected_type}, got {type(data[key])}")

        elif isinstance(data, pd.DataFrame):
            for col, expected_type in self.validation_schema.items():
                if col not in data.columns:
                    logger.warning(f"Schema validation failed: missing column '{col}'")
                    raise ValueError(f"Missing column in DataFrame: {col}")
                # Pandas dtype check is different (simplified here)
                logger.debug(f"Column '{col}' exists, skipping dtype validation for now.")

        else:
            logger.warning("Unsupported data type for schema validation.")
            raise TypeError("Schema validation not supported for this data type.")

        logger.info("Schema validation passed.")

    @lru_cache(maxsize=32)
    def cached_load(self, file_path: str, preprocess_fn: Optional[Callable[[Any], Any]] = None) -> Any:
        """
        Cached version of load() to optimize repeated data loads.
        
        Useful for large, infrequently changing datasets.
        """
        logger.info(f"Loading (cached) {file_path}")
        return self.load(file_path, preprocess_fn)

    @staticmethod
    def batch_load(file_paths: List[str], preprocess_fn: Optional[Callable[[Any], Any]] = None) -> List[Any]:
        """
        Batch load multiple files at once.
        
        Parameters:
        - file_paths (List[str]): List of file paths.
        - preprocess_fn (Callable): Function to apply to each data item.

        Returns:
        - List of loaded datasets.
        """
        logger.info(f"Batch loading {len(file_paths)} files...")

        results = []
        loader = DataLoader()  # independent instance for each batch

        for file in file_paths:
            try:
                result = loader.load(file, preprocess_fn)
                results.append(result)
                logger.info(f"Loaded {file} successfully.")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")

        return results
