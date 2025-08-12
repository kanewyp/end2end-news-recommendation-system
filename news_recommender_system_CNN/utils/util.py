import yaml
import sys
from news_recommender_system_CNN.exception.exception_handler import AppException


def read_yaml_file(file_path: str) -> dict:
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
    

def text_to_sequence(text_tokens, word_dict, max_len=30):
    """Convert tokenized text to sequence of word indices"""
    sequence = []
    for word in text_tokens[:max_len]:  # Truncate to max_len
        if word in word_dict:
            sequence.append(word_dict[word])
        else:
            sequence.append(0)  # Unknown word
    
    # Pad sequence to max_len
    while len(sequence) < max_len:
        sequence.append(0)
    
    return sequence