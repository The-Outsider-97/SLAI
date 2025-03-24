import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """
    Loads dataset from a CSV file.
    
    Args:
        file_path (str): Path to dataset file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
