import yaml
import sys
from news_recommender_system_CNN.exception.exception_handler import AppException


def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    Args:
        file_path (str): The path to the YAML file.
    Returns:
        dict: The content of the YAML file as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise AppException(f"Error reading YAML file {file_path}: {e}", sys) from e