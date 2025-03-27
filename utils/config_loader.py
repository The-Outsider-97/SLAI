import os
import yaml

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML config file.
    
    Returns:
        dict: Parsed configuration dictionary.
    
    Raises:
        FileNotFoundError: If the YAML file is missing.
        yaml.YAMLError: If there's an error in the YAML syntax.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

    config_path = os.path.abspath(config_path)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse config.yaml: {str(e)}")
