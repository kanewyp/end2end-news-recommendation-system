import os
import sys
import ast 
import nltk
import pandas as pd
import pickle
from collections import Counter
from sympy import re
from tqdm import tqdm
import re

from news_recommender_system_CNN.logger.log import logging
from news_recommender_system_CNN.config.configuration import AppConfiguration
from news_recommender_system_CNN.exception.exception_handler import AppException



class DataValidation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def word_tokenize(self, sent):
        # treat consecutive words as words
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []
    
    def preprocess_data(self):
        try:
            news = pd.read_table(self.data_validation_config.news_tsv_file, sep='\t',
                     names=['News_ID', 'Category', 'SubCategory', 'Title',
                                'Abstract','URL','Title_Entities', 'Abstract_Entities', 
                                'Clicked_Count', 'Ignored_Count', 'Total_Appearances', 'Click_Rate'],
                     usecols=['News_ID','Category', 'SubCategory', 'Title','Abstract'],
                     header=0)
            
            behaviours = pd.read_csv(self.data_validation_config.behaviours_tsv_file, sep='\t')
            
            logging.info(f" Shape of news data file: {news.shape}")
            logging.info(f" Shape of behaviours data file: {behaviours.shape}")

            # preprocess data in titles and abstract by lowercasing and removing punctuation
            news['Title'] = news['Title'].str.lower().str.replace('[^\w\s]', '', regex=True)
            news['Abstract'] = news['Abstract'].str.lower().str.replace('[^\w\s]', '', regex=True)
            logging.info(f"lowercased and removed punctuation from Title and Abstract")

            # remove stopwords
            nltk.download('stopwords', quiet=True)
            stopwords = set(nltk.corpus.stopwords.words('english'))
            news['Title'] = news['Title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
            news['Abstract'] = news['Abstract'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
            logging.info(f"removed stopwords from Title and Abstract")
            
            # removing numbers
            news['Title'] = news['Title'].str.replace('\d+', '', regex=True)
            news['Abstract'] = news['Abstract'].str.replace('\d+', '', regex=True)
            logging.info(f"removed numbers from Title and Abstract")

            # indexing the category and subcategory data
            news_category = news['Category'].drop_duplicates().reset_index(drop=True)
            category_dict = {v: k+1 for k, v in news_category.to_dict().items()}
            logging.info(f"Created category mapping")

            news_subcategory = news['SubCategory'].drop_duplicates().reset_index(drop=True)
            subcategory_dict = {v: k+1 for k, v in news_subcategory.to_dict().items()}
            logging.info(f"Created subcategory mapping")

            # tokenize title and abstracts
            news['Title'] = news['Title'].apply(self.word_tokenize)
            news['Abstract'] = news['Abstract'].apply(self.word_tokenize)
            logging.info(f"Tokenized Title and Abstract")

            # create word dictionary
            word_cnt = Counter()
            for i in tqdm(range(len(news))):
                word_cnt.update(news.loc[i]['Title'])
                word_cnt.update(news.loc[i]['Abstract'])
            word_dict = {k: v+1 for k, v in zip(word_cnt, range(len(word_cnt)))}
            logging.info(f"Created word dictionary with {len(word_dict)} unique words")

            # convert user ids into index
            uid2index = {}

            with open(self.data_validation_config.behaviours_tsv_file, 'r') as f:
                for l in tqdm(f):
                    # second column
                    uid = l.strip('\n').split('\t')[1]
                    if uid not in uid2index:
                        uid2index[uid] = len(uid2index) + 1
            logging.info(f"Created user id to index mapping with {len(uid2index)} unique users")

            # store the serialized objects (dictionary)
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            with open(os.path.join(self.data_validation_config.serialized_objects_dir, "word_dict.pkl"), "wb") as f:
                pickle.dump(word_dict, f)
            with open(os.path.join(self.data_validation_config.serialized_objects_dir, "category_dict.pkl"), "wb") as f:
                pickle.dump(category_dict, f)
            with open(os.path.join(self.data_validation_config.serialized_objects_dir, "subcategory_dict.pkl"), "wb") as f:
                pickle.dump(subcategory_dict, f)
            with open(os.path.join(self.data_validation_config.serialized_objects_dir, "uid2index.pkl"), "wb") as f:
                pickle.dump(uid2index, f)
            logging.info(f"Saved serialized objects to {self.data_validation_config.serialized_objects_dir}")
            
            utils_state = {
            'vert_num': len(category_dict),
            'subvert_num': len(subcategory_dict),
            'word_num': len(word_dict),
            'uid2index': len(uid2index)
            }
            logging.info(f"Utils state: {utils_state}")
                        
            # Saving the cleaned data for transformation
            os.makedirs(self.data_validation_config.clean_data_dir, exist_ok=True)
            news.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'news_data.tsv'), index = False)
            behaviours.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'behaviours_data.tsv'), index = False)
            logging.info(f"Saved cleaned data to {self.data_validation_config.clean_data_dir}")


        except Exception as e:
            raise AppException(e, sys) from e

    
    def initiate_data_validation(self):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.preprocess_data()
            logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e



    
        

    