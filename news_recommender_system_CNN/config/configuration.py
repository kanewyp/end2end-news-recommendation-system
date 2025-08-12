import os
import sys
from news_recommender_system_CNN.logger.log import logging
from news_recommender_system_CNN.exception.exception_handler import AppException
from news_recommender_system_CNN.utils.util import read_yaml_file
from news_recommender_system_CNN.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
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
    

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_config = self.configs_info['data_validation_config']
            data_ingestion_config = self.configs_info['data_ingestion_config']
            dataset_dir = data_ingestion_config['dataset_dir']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            news_tsv_file = data_validation_config['news_tsv_file']
            behaviours_tsv_file = data_validation_config['behaviours_tsv_file']

            news_tsv_file_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['ingested_dir'], news_tsv_file)
            behaviours_tsv_file_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['ingested_dir'], behaviours_tsv_file)
            clean_data_path = os.path.join(artifacts_dir, dataset_dir, data_validation_config['clean_data_dir'])
            serialized_objects_dir = os.path.join(artifacts_dir, data_validation_config['serialized_objects_dir'])

            response = DataValidationConfig(
                clean_data_dir = clean_data_path,
                news_tsv_file = news_tsv_file_dir,
                behaviours_tsv_file = behaviours_tsv_file_dir,
                serialized_objects_dir = serialized_objects_dir
            )

            logging.info(f"Data Validation Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e
        

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transformation_config = self.configs_info['data_transformation_config']
            data_validation_config = self.configs_info['data_validation_config']
            data_ingestion_config = self.configs_info['data_ingestion_config']
            dataset_dir = data_ingestion_config['dataset_dir']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
          
            clean_data_file_path = os.path.join(artifacts_dir, dataset_dir, data_validation_config['clean_data_dir'],'clean_data.csv')
            transformed_data_dir = os.path.join(artifacts_dir, dataset_dir, data_transformation_config['transformed_data_dir'])
            glove_url = data_transformation_config['glove_url']
            embedding_dim = data_transformation_config['embedding_dim']

            response = DataTransformationConfig(
                clean_data_file_path = clean_data_file_path,
                transformed_data_dir = transformed_data_dir,
                glove_url = glove_url,
                embedding_dim = embedding_dim
            )

            logging.info(f"Data Transformation Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e
        
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_trainer_config = self.configs_info['model_trainer_config']
            model_transform_config = self.configs_info['data_transformation_config']
            data_ingestion_config = self.configs_info['data_ingestion_config']
            dataset_dir = data_ingestion_config['dataset_dir']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']

            transformed_data_file_dir = os.path.join(artifacts_dir, dataset_dir, model_transform_config['transformed_data_dir'], 'embedding.npy')
            trained_model_dir = os.path.join(artifacts_dir, dataset_dir, model_trainer_config['trained_model_dir'])
            trained_model_name = model_trainer_config['trained_model_name']
            batch_size = model_trainer_config['batch_size']
            num_filters = model_trainer_config['num_filters']
            filter_sizes = model_trainer_config['filter_sizes']
            user_embed_dim = model_trainer_config['user_embed_dim']
            category_embed_dim = model_trainer_config['category_embed_dim']
            dropout = model_trainer_config['dropout']
            learning_rate = model_trainer_config['learning_rate']
            weight_decay = model_trainer_config['weight_decay']
            scheduler_factor = model_trainer_config['scheduler_factor']
            scheduler_patience = model_trainer_config['scheduler_patience']
            patience_stop_criteria = model_trainer_config['patience_stop_criteria']
            num_epochs = model_trainer_config['num_epochs']

            response = ModelTrainerConfig(
                transformed_data_file_dir = transformed_data_file_dir,
                trained_model_dir = trained_model_dir,
                trained_model_name = trained_model_name,
                batch_size = batch_size,
                num_filters = num_filters,
                filter_sizes = filter_sizes,
                user_embed_dim = user_embed_dim,
                category_embed_dim = category_embed_dim,
                dropout = dropout,
                learning_rate = learning_rate,
                weight_decay = weight_decay,
                scheduler_factor = scheduler_factor,
                scheduler_patience = scheduler_patience,
                patience_stop_criteria = patience_stop_criteria,
                num_epochs = num_epochs
            )

            logging.info(f"Model Trainer Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e
