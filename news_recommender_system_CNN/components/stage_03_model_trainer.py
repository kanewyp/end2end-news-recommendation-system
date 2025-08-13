import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from typing import List, Dict, Any

from news_recommender_system_CNN import model
from news_recommender_system_CNN.dataset.dataset_nr import NewsRecommendationDataset
from news_recommender_system_CNN.model.CNN import NewsRecommendationCNN
from news_recommender_system_CNN.logger.log import logging
from news_recommender_system_CNN.config.configuration import AppConfiguration
from news_recommender_system_CNN.exception.exception_handler import AppException


class ModelTrainer:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.model_trainer_config = app_config.get_model_trainer_config()
            self.model_transformation_config = app_config.get_data_transformation_config()
            self.model_validation_config = app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    def load_serialized_objects(self, path) -> dict:
        with open(path, 'rb') as f:
            return pickle.load(f)
        

    def load_embedding_matrix(self, path) -> np.ndarray:
        return np.load(path)
    

    def device_selection(self) -> str:
        # get the python version
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            
            return "xpu"
        else:
            return "cpu"
        

    def load_csv_data(self, path) -> pd.DataFrame:
        return pd.read_csv(path, sep=',')
    
    def load_tsv_data(self, path) -> pd.DataFrame:
        return pd.read_csv(path, sep='\t')


    def prepare_training_data(self, news_data: pd.DataFrame, behaviour_data: pd.DataFrame, uid2index: dict) -> List[Dict[str, Any]]:
        """Prepare training data from news and behaviour data"""
        training_samples = []
    
        # Get user-news interactions
        for _, row in behaviour_data.iterrows():
            user_id = row['User_ID']
            if user_id not in uid2index:
                continue
            
            user_idx = uid2index[user_id]
        
            # Positive samples (clicked news)
            if row['Clicked_News'] and len(row['Clicked_News']) > 0:
                for news_id in row['Clicked_News']:
                    news_info = news_data[news_data['News_ID'] == news_id]
                    if len(news_info) > 0:
                        training_samples.append({
                            'user_idx': user_idx,
                            'news_id': news_id,
                            'label': 1
                        })
        
            # Negative samples (non-clicked news from impressions)
            if row['Non_Clicked_News'] and len(row['Non_Clicked_News']) > 0:
                # Sample some negative examples (to balance dataset)
                neg_samples = row['Non_Clicked_News'][:len(row['Clicked_News'])] if row['Clicked_News'] else row['Non_Clicked_News'][:5]
                for news_id in neg_samples:
                    news_info = news_data[news_data['News_ID'] == news_id]
                    if len(news_info) > 0:
                        training_samples.append({
                            'user_idx': user_idx,
                            'news_id': news_id,
                            'label': 0
                        })
    
        return training_samples


    def train_epoch(self, model, train_loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
    
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            user_idx = batch['user_idx'].to(device)
            title_seq = batch['title_seq'].to(device)
            abstract_seq = batch['abstract_seq'].to(device)
            category_idx = batch['category_idx'].to(device)
            subcategory_idx = batch['subcategory_idx'].to(device)
            labels = batch['label'].to(device)
        
            # Forward pass
            optimizer.zero_grad()
            outputs = model(user_idx, title_seq, abstract_seq, category_idx, subcategory_idx)
            loss = criterion(outputs, labels)
        
            # Backward pass
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
        
            # Store predictions and labels for metrics
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
        avg_loss = total_loss / len(train_loader)
        auc = roc_auc_score(all_labels, all_predictions)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    
        return avg_loss, auc, acc
    

    def evaluate(self,model, val_loader, criterion, device):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
    
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                user_idx = batch['user_idx'].to(device)
                title_seq = batch['title_seq'].to(device)
                abstract_seq = batch['abstract_seq'].to(device)
                category_idx = batch['category_idx'].to(device)
                subcategory_idx = batch['subcategory_idx'].to(device)
                labels = batch['label'].to(device)
            
                # Forward pass
                outputs = model(user_idx, title_seq, abstract_seq, category_idx, subcategory_idx)
                loss = criterion(outputs, labels)
            
                total_loss += loss.item()
            
                # Store predictions and labels for metrics
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
        avg_loss = total_loss / len(val_loader)
        auc = roc_auc_score(all_labels, all_predictions)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    
        return avg_loss, auc, acc


    def train(self):
        try:
            # load serialized objects
            word_dict = self.load_serialized_objects(os.path.join(self.model_validation_config.serialized_objects_dir, "word_dict.pkl"))
            category_dict = self.load_serialized_objects(os.path.join(self.model_validation_config.serialized_objects_dir, "category_dict.pkl"))
            subcategory_dict = self.load_serialized_objects(os.path.join(self.model_validation_config.serialized_objects_dir, "subcategory_dict.pkl"))
            uid2index = self.load_serialized_objects(os.path.join(self.model_validation_config.serialized_objects_dir, "uid2index.pkl"))
            logging.info("Serialized objects loaded successfully.")

            # load embedding matrix
            embedding_matrix = self.load_embedding_matrix(os.path.join(self.model_transformation_config.transformed_data_dir, "embedding.npy"))
            logging.info("Embedding matrix loaded successfully.")

            logging.info(f"Vocabulary size: {len(word_dict)}")
            logging.info(f"Embedding matrix shape: {embedding_matrix.shape}")
            logging.info(f"Number of users: {len(uid2index)}")
            logging.info(f"Number of categories: {len(category_dict)}")
            logging.info(f"Number of subcategories: {len(subcategory_dict)}")

            # set device for pytorch
            device_name = self.device_selection()
            device = torch.device(device_name)
            logging.info(f"{sys.version}")
            logging.info(f"Using device: {device_name}")

            # load data
            news_data = self.load_csv_data(os.path.join(self.model_validation_config.clean_data_dir, "news_data.tsv"))
            behaviours_data = self.load_csv_data(os.path.join(self.model_validation_config.clean_data_dir, "behaviours_data.tsv"))
            logging.info("News and Behaviours data loaded successfully.")

            # convert string representation of lists to actual lists
            for col in ['Clicked_News', 'Non_Clicked_News']:
                behaviours_data[col] = behaviours_data[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            logging.info("News data converted successfully.")

            # prepare training data
            logging.info("Preparing training data...")
            training_samples = self.prepare_training_data(news_data, behaviours_data, uid2index)
            logging.info(f"Total training samples: {len(training_samples)}")

            # train_test_split
            train_samples, val_samples = train_test_split(training_samples, test_size=0.2, random_state=42,stratify=[s['label'] for s in training_samples])
            logging.info(f"Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

            # Create datasets and dataloaders
            train_dataset = NewsRecommendationDataset(train_samples, news_data, word_dict, category_dict, subcategory_dict)
            val_dataset = NewsRecommendationDataset(val_samples, news_data, word_dict, category_dict, subcategory_dict)
            logging.info("Datasets created successfully.")
            train_loader = DataLoader(train_dataset, batch_size=self.model_trainer_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.model_trainer_config.batch_size, shuffle=False)
            logging.info("DataLoaders created successfully.")

            # Initialize model
            model = NewsRecommendationCNN(
                embedding_matrix=embedding_matrix,
                num_users=len(uid2index),
                num_categories=len(category_dict),
                num_subcategories=len(subcategory_dict),
                embed_dim=self.model_transformation_config.embedding_dim,
                num_filters=self.model_trainer_config.num_filters,
                filter_sizes=self.model_trainer_config.filter_sizes,
                user_embed_dim=self.model_trainer_config.user_embed_dim,
                category_embed_dim=self.model_trainer_config.category_embed_dim,
                dropout=self.model_trainer_config.dropout
            ).to(device)
            logging.info("Model initialized successfully.")

            # Initialize training components
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(),
                                   lr=float(self.model_trainer_config.learning_rate),
                                   weight_decay=float(self.model_trainer_config.weight_decay))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             factor=float(self.model_trainer_config.scheduler_factor),
                                                             patience=float(self.model_trainer_config.scheduler_patience))
            logging.info("Training components initialized successfully.")

            # Training Loop
            num_epochs = int(self.model_trainer_config.num_epochs)
            patience_stop_criteria = int(self.model_trainer_config.patience_stop_criteria)
            patience_counter = 0
            best_model_state = None
            best_val_auc = 0

            logging.info("Starting training...")
            logging.info(f"Training for {num_epochs} epochs")
            logging.info("-" * 60)

            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}")
                logging.info(f"Epoch {epoch + 1}/{num_epochs}")

                # Training
                train_loss, train_auc, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, device)
    
                # Validation
                val_loss, val_auc, val_acc = self.evaluate(model, val_loader, criterion, device)

                # Learning rate scheduling
                scheduler.step(val_loss)
    
                # Print metrics
                print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                logging.info(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f}")
                logging.info(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
                logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

                # Save best model
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    print(f"New best model! AUC: {best_val_auc:.4f}")
                    logging.info(f"New best model! AUC: {best_val_auc:.4f}")
                else:
                    patience_counter += 1
    
                # Early stopping
                if patience_counter >= patience_stop_criteria:
                    print(f"Early stopping triggered after {patience_stop_criteria} epochs without improvement")
                    logging.info(f"Early stopping triggered after {patience_stop_criteria} epochs without improvement")
                    break

                print("-" * 60)
                logging.info("-" * 60)
            
            # Load best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"Loaded best model with validation AUC: {best_val_auc:.4f}")

                # Save the trained model
                os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'word_dict': word_dict,
                    'category_dict': category_dict,
                    'subcategory_dict': subcategory_dict,
                    'uid2index': uid2index,
                    'best_val_auc': best_val_auc,
                    'model_config': {
                        'embed_dim': self.model_transformation_config.embedding_dim,
                        'num_filters': self.model_trainer_config.num_filters,
                        'filter_sizes': self.model_trainer_config.filter_sizes,
                        'user_embed_dim': self.model_trainer_config.user_embed_dim,
                        'category_embed_dim': self.model_trainer_config.category_embed_dim,
                        'dropout': self.model_trainer_config.dropout
                    }
                }, os.path.join(self.model_trainer_config.trained_model_dir,
                                 self.model_trainer_config.trained_model_name))

                print("Training completed!")
                print(f"Model saved as '{self.model_trainer_config.trained_model_name}'")
                print(f"Best validation AUC: {best_val_auc:.4f}")
                logging.info("Training completed")
                logging.info(f"Model saved as 'cnn_model.pth'")
                logging.info(f"Best validation AUC: {best_val_auc:.4f}")

        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_model_trainer(self):
        try:
            logging.info(f"{'='*20}Model Trainer log started.{'='*20} ")
            self.train()
            logging.info(f"{'='*20}Model Trainer log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e