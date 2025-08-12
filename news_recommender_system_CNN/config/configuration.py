import os
import sys
from news_recommender_system_CNN.logger.log import logging
from news_recommender_system_CNN.exception.exception_handler import AppException
from news_recommender_system_CNN.utils.util import read_yaml_file
from news_recommender_system_CNN.entity.config_entity import DataIngestionConfig
from news_recommender_system_CNN.constant import *

class AppConfiguration:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH):
        try:
            self.configs_info = read_yaml_file(file_path=config_file_path)
        except Exception as e:
            raise AppException(e, sys) from e

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_config = self.configs_info['data_ingestion_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            dataset_dir = data_ingestion_config['dataset_dir']

            ingested_data_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['ingested_dir'])
            raw_data_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['raw_data_dir'])

            response = DataIngestionConfig(
                dataset_download_url = data_ingestion_config['dataset_download_url'],
                raw_data_dir = raw_data_dir,
                ingested_data_dir = ingested_data_dir
            )

            logging.info(f"Data Ingestion Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e