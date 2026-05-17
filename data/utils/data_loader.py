from __future__ import annotations

import os
import json
import yaml
import csv
import pickle
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import re
import io
import html
import time

from typing import Any, Dict, List, Optional, Callable, Union
from functools import lru_cache
from pathlib import Path

from src.utils.system_optimizer import SystemOptimizer
from logs.logger import get_logger

SUPPORTED_FORMATS = ["json", "yaml", "csv", "parquet", "pickle"]

logger = get_logger("Data Loader")

# Add SecurityError definition
class SecurityError(Exception):
    """Exception for security policy violations"""
    pass

# system_metrics placeholder
system_metrics = {}  # Placeholder for actual metrics implementation

class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    pass

class SafeDataLoader:
    """Secure data loader with Ferreres-inspired sanitization framework"""
    
    def __init__(self, validation_schema: Optional[Dict] = None, secure_mode: bool = True):
        """
        Initialize with validation rules and security constraints
        
        Args:
            validation_schema: Nested dictionary specifying data constraints
            secure_mode: Disables risky formats like pickle when True (default)
        """
        self.validation_schema = validation_schema or {}
        self.secure_mode = secure_mode
        self._init_builtin_preprocessors()
        logger.info("Initialized SafeDataLoader with Ferreres sanitization protocols")

    def load(self, file_path: str, preprocessors: Optional[List[str]] = None) -> Any:
        """
        Secure loading pipeline with validation and sanitization
        
        1. Format detection
        2. Schema validation
        3. Data sanitization
        4. Preprocessing
        """
        self._validate_file_path(file_path)
        file_format = self._detect_format(file_path)
        load_fn = getattr(self, f"_load_{file_format}", None)
        
        if not load_fn:
            raise ValueError(f"Unsupported format: {file_format}")
            
        raw_data = load_fn(file_path)
        self._validate_schema(raw_data)
        sanitized = self._sanitize_data(raw_data)
        
        if preprocessors:
            for p in preprocessors:
                sanitized = self._apply_preprocessor(sanitized, p)
                
        return sanitized

    def _validate_file_path(self, file_path: str) -> None:
        """Security checks for file access"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if os.path.islink(file_path):
            raise SecurityError("Symbolic links are not allowed for security reasons")

    def _detect_format(self, file_path: str) -> str:
        """Multi-layered format detection with content analysis"""
        ext = Path(file_path).suffix[1:].lower()
        
        # Content-based validation
        with open(file_path, 'rb') as f:
            header = f.read(4)
            
        if ext == 'parquet' and header == b'PAR1':
            return 'parquet'
        if ext == 'csv' and self._is_valid_csv(f):
            return 'csv'
        if ext == 'pickle' and self.secure_mode:
            raise SecurityError("Pickle loading disabled in secure mode")
            
        return ext  # Fallback to extension-based

    def _is_valid_csv(self, file_handle) -> bool:
        """Basic CSV validation checking for consistent column counts"""
        try:
            # Reset file pointer after header read
            file_handle.seek(0)
            reader = csv.reader(io.TextIOWrapper(file_handle, encoding='utf-8'))
            header = next(reader)
            for row in reader:
                if len(row) != len(header):
                    return False
            return True
        except Exception:
            return False

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Secure CSV loading with malformed header handling"""
        try:
            # Read CSV with manual header correction
            df = pd.read_csv(
                file_path,
                delimiter=';',
                skiprows=1,  # Skip the "Sanitized data:" header line
                header=None,
                names=['index', 'name', 'email', 'age'],  # Define explicit column names
                usecols=['name', 'email', 'age'],  # Ignore the index column
                skipinitialspace=True  # Clean whitespace
            )
            logger.info(f"Loaded CSV with valid columns: {df.columns.tolist()}")
            return df
        except pd.errors.ParserError as e:
            raise DataValidationError(f"CSV parsing error: {str(e)}")

    def _load_parquet(self, file_path: str) -> pd.DataFrame:
        """Parquet loading with schema validation"""
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            # Validate parquet schema
            if self.validation_schema:
                for col, meta in self.validation_schema.items():
                    if col not in table.schema.names:
                        raise DataValidationError(f"Missing required column: {col}")
            return df
        except Exception as e:
            raise DataValidationError(f"Parquet error: {str(e)}")

    def _validate_schema(self, data: Union[Dict, pd.DataFrame]) -> None:
        """Multi-level schema validation"""
        if isinstance(data, dict):
            self._validate_dict_schema(data)
        elif isinstance(data, pd.DataFrame):
            self._validate_df_schema(data)
        else:
            logger.warning("Schema validation skipped for unsupported type")

    def _validate_dict_schema(self, data: dict) -> None:
        """Comprehensive dictionary schema validation"""
        for key, constraints in self.validation_schema.items():
            if key not in data:
                raise DataValidationError(f"Missing required key: {key}")
            
            value = data[key]
            # Type validation
            if 'type' in constraints and not isinstance(value, constraints['type']):
                raise DataValidationError(
                    f"Type mismatch for {key}: Expected {constraints['type']}, got {type(value)}"
                )
            
            # Value constraints
            if 'min' in constraints and value < constraints['min']:
                raise DataValidationError(
                    f"Value {value} for {key} below minimum {constraints['min']}"
                )
                
            if 'max' in constraints and value > constraints['max']:
                raise DataValidationError(
                    f"Value {value} for {key} above maximum {constraints['max']}"
                )
            
            # Pattern matching
            if 'regex' in constraints and isinstance(value, str):
                if not re.fullmatch(constraints['regex'], value):
                    raise DataValidationError(
                        f"Value '{value}' for {key} fails regex validation"
                    )

    def _validate_df_schema(self, df: pd.DataFrame) -> None:
        """Advanced DataFrame validation with type checking"""
        df.columns = df.columns.str.lower()
        for col, constraints in self.validation_schema.items():
            if col not in df.columns:
                raise DataValidationError(f"Missing column: {col}")
                
            # Type validation
            if 'type' in constraints:
                if not df[col].apply(lambda x: isinstance(x, constraints['type'])).all():
                    raise DataValidationError(f"Type mismatch in column {col}")

            # Value constraints
            if 'min' in constraints:
                if df[col].min() < constraints['min']:
                    raise DataValidationError(f"Value below min in {col}")
                    
            if 'regex' in constraints:
                pattern = re.compile(constraints['regex'])
                if not df[col].astype(str).str.match(pattern).all():
                    raise DataValidationError(f"Pattern mismatch in {col}")

    def _sanitize_data(self, data: Any) -> Any:
        """Core sanitization pipeline"""
        if isinstance(data, pd.DataFrame):
            return self._sanitize_dataframe(data)
        elif isinstance(data, dict):
            return self._sanitize_dict(data)
        return data

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame-specific sanitization"""
        # Remove NaN values
        df = df.dropna()

        # Escape potential HTML/JS
        str_cols = df.select_dtypes(include=['object']).columns
        df[str_cols] = df[str_cols].applymap(lambda x: html.escape(x) if isinstance(x, str) else x)
        
        return df

    def _sanitize_dict(self, data: dict) -> dict:
        """Recursive dictionary sanitization with HTML escaping"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = html.escape(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_data(item) for item in value]
            else:
                sanitized[key] = value
        return sanitized    

    def _init_builtin_preprocessors(self) -> None:
        """Register built-in preprocessing functions"""
        self.preprocessors = {
            'normalize': self._normalize,
            'fill_na': self._fill_missing,
            'encode_categorical': self._encode_categorical
        }

    def _apply_preprocessor(self, data: Any, name: str) -> Any:
        """Apply named preprocessing step"""
        if name not in self.preprocessors:
            raise ValueError(f"Unknown preprocessor: {name}")
        return self.preprocessors[name](data)

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization for numerical columns"""
        num_cols = data.select_dtypes(include=np.number).columns
        data[num_cols] = (data[num_cols] - data[num_cols].mean()) / data[num_cols].std()
        return data

    def _fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing value handling"""
        # Numerical: median imputation
        num_cols = data.select_dtypes(include=np.number).columns
        data[num_cols] = data[num_cols].fillna(data[num_cols].median())
        
        # Categorical: mode imputation
        cat_cols = data.select_dtypes(exclude=np.number).columns
        data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
        
        return data

    def _encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Safe one-hot encoding implementation"""
        return pd.get_dummies(data, drop_first=True)
# ============================================
# Flexible Data Loader Class
# ============================================
class FlexibleDataLoader:
    """
    Flexible DataLoader that supports JSON, YAML, CSV, Parquet, Pickle, and Pandas DataFrames.
    Provides optional schema validation, preprocessing, caching, and batch loading.
    """

    def __init__(self, validation_schema=None, optimizer: SystemOptimizer = None):
        self.validation_schema = validation_schema or {}
        logger.info(f"FlexibleDataLoader initialized with schema: {self.validation_schema}")

        self.optimizer = optimizer
        self.adaptive_batch_size = None
       
    def load(self, file_path: str, preprocess_fn: Optional[Callable[[Any], Any]] = None) -> Any:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1][1:].lower()
        logger.info(f"Loading file '{file_path}' with extension '{ext}'")
        
        if self.optimizer:
            self.adaptive_batch_size = self.optimizer.dynamic_batch_calculator(
                current_metrics=system_metrics  # using the defined placeholder
            )

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
        return self._process_data(data, preprocess_fn)

    def _process_data(self, data: Any, preprocess_fn: Optional[Callable[[Any], Any]] = None) -> Any:
        logger.info("Running final post-processing stage...")

        # Convert DataFrame to list of dicts if needed (standard output format)
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
            logger.debug("Converted DataFrame to list of records.")

        # Optional: Inject audit tag (timestamp, source path, etc.)
        audit_tag = {"processed_at": pd.Timestamp.now().isoformat()}
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    row.update(audit_tag)

        # Apply optional postprocessing function
        if preprocess_fn:
            try:
                logger.debug("Applying user-defined post-processing function...")
                data = preprocess_fn(data)
            except Exception as e:
                logger.warning(f"Post-processing function raised error: {e}")

        logger.info("Post-processing complete.")
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

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Secure CSV loading with malformed header handling"""
        try:
            # Read CSV with manual header correction
            df = pd.read_csv(
                file_path,
                delimiter=';',
                skiprows=1,  # Skip the "Sanitized data:" header line
                header=None,
                names=['index', 'name', 'email', 'age'],  # Define explicit column names
                usecols=['name', 'email', 'age'],  # Ignore the index column
                skipinitialspace=True  # Clean whitespace
            )
            logger.info(f"Loaded CSV with valid columns: {df.columns.tolist()}")
            return df
        except pd.errors.ParserError as e:
            raise DataValidationError(f"CSV parsing error: {str(e)}")

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

# Example usage
if __name__ == "__main__":
    # Define validation schema
    schema = {
        "name": {"type": str},
        "age": {"type": int, "min": 12, "max": 150},
        "email": {"type": str, "regex": r"^[^@]+@[^@]+\.[^@]+$"}
    }

    # Test 1: Normal operation with valid data
    print("\n=== Test 1: Basic Valid Load ===")
    try:
        loader = SafeDataLoader(validation_schema=schema)
        data = loader.load(
            "data/users.csv",
            preprocessors=["fill_na", "encode_categorical"]
        )
        print("Sanitized data sample:")
        print(data.head())
        print("\nData types:")
        print(data.dtypes)
    except Exception as e:
        print(f"Test 1 Failed: {str(e)}")

    # Test 2: Invalid schema validation
    print("\n=== Test 2: Schema Validation Failure ===")
    try:
        bad_schema = schema.copy()
        bad_schema["phone"] = {"type": str}
        loader = SafeDataLoader(validation_schema=bad_schema)
        loader.load("data/users.csv")
    except DataValidationError as e:
        print(f"Expected validation error caught: {str(e)}")

    # Test 3: Invalid file path
    print("\n=== Test 3: File Not Found ===")
    try:
        loader = SafeDataLoader()
        loader.load("data/nonexistent.csv")
    except FileNotFoundError as e:
        print(f"Expected error caught: {str(e)}")

    # Test 4: Malformed CSV
    print("\n=== Test 4: Malformed CSV Handling ===")
    try:
        loader = SafeDataLoader(schema)
        loader.load("data/broken.csv")
    except DataValidationError as e:
        print(f"CSV parsing error handled: {str(e)}")

    # Test 5: Security validation
    print("\n=== Test 5: Security Checks ===")
    try:
        # This test requires creating a symlink for demonstration
        loader = SafeDataLoader()
        loader.load("data/symlink.csv")
    except SecurityError as e:
        print(f"Security violation caught: {str(e)}")

    # Test 6: Full pipeline with FlexibleDataLoader
    print("\n=== Test 6: FlexibleDataLoader with Custom Processing ===")
    try:
        def custom_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
            """Add year of birth column"""
            current_year = pd.Timestamp.now().year
            df["yob"] = current_year - df["age"]
            return df

        flex_loader = FlexibleDataLoader(validation_schema=schema)
        data = flex_loader.load(
            "data/users.csv",
            preprocess_fn=custom_preprocessor
        )
        print("\nData with calculated year of birth:")
        print(data)
    except Exception as e:
        print(f"Flexible loader failed: {str(e)}")

    # Test 7: Type validation failure
    print("\n=== Test 7: Type Validation ===")
    try:
        type_schema = schema.copy()
        type_schema["age"]["type"] = str  # Intentional type mismatch
        loader = SafeDataLoader(type_schema)
        loader.load("data/users.csv")
    except DataValidationError as e:
        print(f"Type validation worked: {str(e)}")

    # Test 8: Performance testing
    print("\n=== Test 8: Cached Loading ===")
    try:
        flex_loader = FlexibleDataLoader()
        start = time.time()
        data1 = flex_loader.cached_load("data/users.csv")
        first_load = time.time() - start
        
        start = time.time()
        data2 = flex_loader.cached_load("data/users.csv")
        cached_load = time.time() - start
        
        print(f"First load: {first_load:.4f}s, Cached load: {cached_load:.4f}s")
        print("Cache validation:", data1.equals(data2))
    except Exception as e:
        print(f"Cache test failed: {str(e)}")
