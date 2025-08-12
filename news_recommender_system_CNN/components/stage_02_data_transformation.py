import os
import sys
import pickle
import zipfile
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
from tqdm import tqdm
import requests

from news_recommender_system_CNN.logger.log import logging
from news_recommender_system_CNN.config.configuration import AppConfiguration
from news_recommender_system_CNN.exception.exception_handler import AppException


class DataTransformation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_transformation_config = app_config.get_data_transformation_config()
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    def create_embeddings_with_glove(self, embedding_dim):
        # create temp directory to store pre-trained glove moembedding
        tmpdir = TemporaryDirectory()
        data_path = tmpdir.name
        logging.info(f"Temporary directory created at: {data_path}")

        # Download GloVe embeddings
        glove_url = self.data_transformation_config.glove_url
        glove_zip_path = os.path.join(data_path,"glove.6B.zip")
        glove_dir = os.path.join(data_path,"glove")

        # Download the zip file if it does not exist
        if not os.path.exists(glove_zip_path):
            logging.info("Downloading GloVe embeddings from Hugging Face...")
            with requests.get(glove_url, stream=True) as r:
                r.raise_for_status()
                with open(glove_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logging.info("Download complete.")
        else:
            logging.info("GloVe zip file already exists.")

        # Extract the zip file if not already extracted
        if not os.path.exists(glove_dir):
            logging.info("Extracting GloVe embeddings...")
            with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
                zip_ref.extractall(glove_dir)
            logging.info("Extraction complete.")
        else:
            logging.info("GloVe directory already exists.")


        # get word_dict from serialized_object
        with open(self.data_validation_config.serialized_objects_dir + "/word_dict.pkl", "rb") as f:
            word_dict = pickle.load(f)
        logging.info(f"Word dictionary loaded with {len(word_dict)} words.")

        # create embedding matrix
        embedding_matrix = np.zeros((len(word_dict) + 1, embedding_dim))
        exist_word = []

        with open(f"{data_path}/glove/glove.6B.{embedding_dim}d.txt", "rb") as f:
            for l in tqdm(f):  # noqa: E741 ambiguous variable name 'l'
                l = l.split()  # noqa: E741 ambiguous variable name 'l'
                word = l[0].decode()
                if len(word) != 0:
                    if word in word_dict:
                        wordvec = [float(x) for x in l[1:]]
                        index = word_dict[word]
                        embedding_matrix[index] = np.array(wordvec)
                        exist_word.append(word)
        logging.info(f"Embedding matrix created")

        # get the matrix dimension
        self.embedding_dim = embedding_matrix.shape[1]
        self.vocab_size = embedding_matrix.shape[0]
        logging.info(f"Embedding matrix dimensions: {self.vocab_size} x {self.embedding_dim}")

        # create transformed_data_dir
        os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
        np.save(os.path.join(self.data_transformation_config.transformed_data_dir, "embedding.npy"), embedding_matrix)
        logging.info(f"Saved embedding matrix to {self.data_transformation_config.transformed_data_dir}")

        # clean up the temporary files
        logging.info(f"Temporary directory {data_path} cleaned up.")
        tmpdir.cleanup()


    def get_data_transformer(self):
        try:
            self.create_embeddings_with_glove(self.data_transformation_config.embedding_dim)
        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20} ")
            self.get_data_transformer()
            logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e