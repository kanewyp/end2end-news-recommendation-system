import os
import logging
from pathlib import Path # for handling file paths

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] :  %(message)s : ')

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/pipeline/__init__.py",
    "src/entity/__init__.py",
    "src/constants/__init__.py",
    "src/constants/constant.py",
    "data/description.txt",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

for file_path in list_of_files:
    file_path = Path(file_path)
    if not file_path.exists():
        file_dir = file_path.parent
        if not file_dir.exists():
            logging.info(f"Creating directory: {file_dir}")
            os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating file: {file_path}")
        with open(file_path, 'w') as f:
            pass  # Create an empty file