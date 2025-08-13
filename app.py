import os
import sys
import pickle
import pandas as pd

from torch import device
import torch
import streamlit as st
import numpy as np

from news_recommender_system_CNN.logger.log import logging
from news_recommender_system_CNN.config.configuration import AppConfiguration
from news_recommender_system_CNN.pipeline.training_pipeline import TrainingPipeline
from news_recommender_system_CNN.exception.exception_handler import AppException
from news_recommender_system_CNN.model.CNN import NewsRecommendationCNN


class Recommendation:
    def __init__(self,app_config = AppConfiguration()):
        try:
            self.recommendation_config= app_config.get_recommendation_config()
            self.model_trainer_config = app_config.get_model_trainer_config()
            self.model_transformation_config = app_config.get_data_transformation_config()
            self.model_validation_config = app_config.get_data_validation_config()
            self.model_ingestion_config = app_config.get_data_ingestion_config()
            self.news_data = pd.read_csv(self.model_validation_config.news_tsv_file, sep="\t")
            self.model_path = os.path.join(self.model_trainer_config.trained_model_dir, self.model_trainer_config.trained_model_name)
            self.user_ids = self.load_serialized_objects("uid2index.pkl")
            self.word_dict = self.load_serialized_objects("word_dict.pkl")
            self.category_dict = self.load_serialized_objects("category_dict.pkl")
            self.subcategory_dict = self.load_serialized_objects("subcategory_dict.pkl")
            self.uid2index = self.load_serialized_objects("uid2index.pkl")
            self.embedding_matrix = np.load(os.path.join(self.model_transformation_config.transformed_data_dir, "embedding.npy"))
            self.candidates = self.prepare_news_as_candidates(self.news_data)
        except Exception as e:
            raise AppException(e, sys) from e


    def fetch_url(self, news_id):
        try:
            news_row = self.news_data[self.news_data['News_ID'] == news_id]
            if not news_row.empty:
                return news_row.iloc[0]['URL']
        except Exception as e:
            raise AppException(e, sys) from e
        return None


    def user_exists(self, user_id):
        try:
            return user_id in user_ids
        except Exception as e:
            raise AppException(e, sys) from e
        

    def load_serialized_objects(self, file_name):
        try:
            with open(os.path.join(self.model_validation_config.serialized_objects_dir, file_name), "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise AppException(e, sys) from e
    

    def text_to_sequence(self, text_tokens, word_dict, max_len=30):
        try:
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
        
        except Exception as e:
            raise AppException(e, sys) from e


    def get_user_list(self):
        uid2index = self.load_serialized_objects("uid2index.pkl")
        return list(uid2index.keys())
    
    def prepare_news_as_candidates(self, news_data):
        # Prepare all news articles as candidates
        candidates = []
        for _, news_row in news_data.iterrows():
            candidates.append({
                'news_id': news_row['News_ID'],
                'title': news_row['Title'],
                'category': news_row['Category'],
                'subcategory': news_row['SubCategory'],
                'title_seq': self.text_to_sequence(news_row['Title'], self.word_dict, max_len=30),
                'abstract_seq': self.text_to_sequence(news_row['Abstract'], self.word_dict, max_len=50),
                'category_idx': self.category_dict.get(news_row['Category'], 0),
                'subcategory_idx': self.subcategory_dict.get(news_row['SubCategory'], 0),
                'url': self.fetch_url(news_row['News_ID'])
            })
        return candidates
            


    def recommend_for_user(self, user_id, device="cpu"):
        try:
            top_k = self.recommendation_config.top_k

            # Check if user exists
            if not self.user_exists(user_id):
                logging.info(f"User_ID {user_id} not found in the dataset.")
                return []

            # Load the trained model
            logging.info(f"Loading model from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            model_config = checkpoint['model_config']

            # Initialize model with saved configuration
            model = NewsRecommendationCNN(
                embedding_matrix=self.embedding_matrix,
                num_users=len(self.uid2index),
                num_categories=len(self.category_dict),
                num_subcategories=len(self.subcategory_dict),
                **model_config
            ).to(device)

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            user_idx = self.uid2index[user_id]

            
            # Batch inference for efficiency
            batch_size = 128
            scores = []
            candidates = self.candidates
            logging.info(f"Total candidates: {len(candidates)}. Processing in batches of size {batch_size}...")
            with torch.no_grad():
                for i in range(0, len(candidates), batch_size):
                    batch = candidates[i:i+batch_size]
                    batch_size_actual = len(batch)

                    # Prepare batch tensors
                    user_idxs = torch.tensor([user_idx] * batch_size_actual, dtype=torch.long, device=device)
                    title_seqs = torch.tensor([c['title_seq'] for c in batch], dtype=torch.long, device=device)
                    abstract_seqs = torch.tensor([c['abstract_seq'] for c in batch], dtype=torch.long, device=device)
                    category_idxs = torch.tensor([c['category_idx'] for c in batch], dtype=torch.long, device=device)
                    subcategory_idxs = torch.tensor([c['subcategory_idx'] for c in batch], dtype=torch.long, device=device)

                    # Get predictions
                    outputs = model(user_idxs, title_seqs, abstract_seqs, category_idxs, subcategory_idxs)
                    scores.extend(outputs.cpu().numpy())

            # Combine news with their scores and sort by score (descending)
            news_with_scores = []
            url = []
            for i, candidate in enumerate(candidates):
                news_with_scores.append({
                    'news_id': candidate['news_id'],
                    'title': candidate['title'],
                    'category': candidate['category'],
                    'subcategory': candidate['subcategory'],
                    'score': scores[i]
                })
                url.append(candidate['url'])

            # Sort by score in descending order
            news_with_scores.sort(key=lambda x: x['score'], reverse=True)

            # Display top recommendations
            print(f"\n" + "="*80)
            print(f"TOP {top_k} NEWS RECOMMENDATIONS FOR USER: {user_id}")
            print("="*80)


            for i, news in enumerate(news_with_scores[:top_k], 1):
                print(f"{i:2d}. Score: {news['score']:.4f} | {news['category']} > {news['subcategory']}")
                print(f"    News ID: {news['news_id']}")
                print(f"    Title: {news['title']}")
                logging.info(f"{i:2d}. Score: {news['score']:.4f} | {news['category']} > {news['subcategory']}")
                logging.info(f"    News ID: {news['news_id']}")
                logging.info(f"    Title: {news['title']}")
                print("-" * 80)
                logging.info("-" * 80)

            return news_with_scores[:top_k], url[:top_k]
        
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    
    def recommendations_engine(self, user_id):
        try:
            recommended_news, news_url = self.recommend_for_user(user_id)
        
            # Create a header for the recommendations section in MAIN AREA
            st.markdown("---")
            st.subheader(f"🎯 Top 5 Recommendations for User: {user_id}")
            st.markdown("---")

            # Display each recommendation in its own row in MAIN AREA
            for i, (news, url) in enumerate(zip(recommended_news[:5], news_url[:5]), 1):
                # Create a container for each news article
                with st.container():
                    # Main title with ranking
                    st.markdown(f"### {i}️⃣ {news['title']}")

                    # Create single-level columns for info
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"**🏷️ Category:**  \n{news['category'].capitalize()}")
                    with col2:
                        st.markdown(f"**🔖 Subcategory:**  \n{news['subcategory']}")
                    with col3:
                        st.markdown(f"**⭐ Score:**  \n{news['score']:.3f}")
                    with col4:
                        # Read more button
                        if url and url != "None":
                            st.link_button("📖 Read More", url)
                        else:
                            st.info("No URL available")

                    # Add abstract if available
                    news_row = self.news_data[self.news_data['News_ID'] == news['news_id']]
                    if not news_row.empty and 'Abstract' in news_row.columns:
                        abstract = news_row.iloc[0]['Abstract']
                        if abstract and str(abstract) != 'nan':
                            st.markdown(f"**📝 Abstract:** {abstract}")

                    # News ID in smaller text
                    st.caption(f"News ID: {news['news_id']}")

                    # Add separator between articles
                    if i < 5:
                        st.markdown("---")
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            raise AppException(e, sys) from e



if __name__ == "__main__":
    # Set page configuration for better appearance
    st.set_page_config(
        page_title="News Recommender System",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main header with styling
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4; margin-bottom: 30px;'>
            📰 News Recommender System
        </h1>
        <p style='text-align: center; font-size: 18px; color: #666; margin-bottom: 40px;'>
            CNN-based Personalized News Recommendation Engine
        </p>
    """, unsafe_allow_html=True)

    # Initialize the recommendation object
    try:
        obj = Recommendation()
        st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.stop()

    # Create sidebar for controls ONLY
    with st.sidebar:
        st.header("🛠️ Controls")
        
        # Training section
        st.subheader("1. Model Training")
        if st.button('🚀 Train News System', type="primary"):
            with st.spinner("Training in progress..."):
                obj.train_engine()
        
        st.markdown("---")
        
        # User selection section
        st.subheader("2. User Selection")
        user_ids = obj.get_user_list()
        selected_user = st.selectbox(
            "👤 Select a User ID:",
            options=user_ids[1:],  # Skip the first item if it's a header
            help="Choose a user to get personalized recommendations"
        )
        
        # Recommendation button
        get_recommendations = st.button('🎯 Show Recommendations', type="primary")

    # MAIN CONTENT AREA - where recommendations will appear
    if get_recommendations and selected_user:
        with st.spinner("Generating recommendations..."):
            obj.recommendations_engine(selected_user)  # This will display in main area
    elif get_recommendations and not selected_user:
        st.warning("Please select a user first!")
    else:
        # Welcome message when no recommendations are being shown
        st.markdown("""
            <div style='text-align: center; padding: 50px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;'>
                <h3>🌟 Welcome to the News Recommendation System!</h3>
                <p>Use the sidebar to train the model and get personalized news recommendations.</p>
                <p><strong>Steps:</strong></p>
                <ol style='text-align: left; display: inline-block;'>
                    <li>🚀 Train the system (if not already trained)</li>
                    <li>👤 Select a user from the dropdown</li>
                    <li>🎯 Click "Show Recommendations" to see personalized news</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)