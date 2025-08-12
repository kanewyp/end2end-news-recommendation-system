from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig", [
    "dataset_download_url",
    "raw_data_dir",
    "ingested_data_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["clean_data_dir",
                                                         "news_tsv_file",
                                                         "behaviours_tsv_file",
                                                         "serialized_objects_dir"])     

DataTransformationConfig = namedtuple("DataTransformationConfig", ["clean_data_file_path",
                                                                   "glove_url",
                                                                   "transformed_data_dir",
                                                                   "embedding_dim"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["transformed_data_file_dir",
                                                      "trained_model_dir",
                                                      "trained_model_name",
                                                      "batch_size",
                                                      "num_filters",
                                                      "filter_sizes",
                                                      "user_embed_dim",
                                                      "category_embed_dim",
                                                      "dropout",
                                                      "learning_rate",
                                                      "weight_decay",
                                                      "scheduler_factor",
                                                      "scheduler_patience",
                                                      "patience_stop_criteria",
                                                      "num_epochs"
                                                      ])

