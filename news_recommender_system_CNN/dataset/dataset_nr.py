import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from news_recommender_system_CNN.utils.util import text_to_sequence


# Dataset class for news recommendation
class NewsRecommendationDataset(Dataset):
    def __init__(self, samples, news_data, word_dict, category_dict, subcategory_dict, max_title_len=30, max_abstract_len=50):
        self.samples = samples
        self.news_data = news_data.set_index('News_ID')
        self.word_dict = word_dict
        self.category_dict = category_dict
        self.subcategory_dict = subcategory_dict
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        news_id = sample['news_id']
        user_idx = sample['user_idx']
        label = sample['label']
        
        # Get news information
        try:
            news_row = self.news_data.loc[news_id]
            
            # Convert title and abstract to sequences
            title_seq = text_to_sequence(news_row['Title'], self.word_dict, self.max_title_len)
            abstract_seq = text_to_sequence(news_row['Abstract'], self.word_dict, self.max_abstract_len)
            
            # Get category and subcategory indices
            category_idx = self.category_dict.get(news_row['Category'], 0)
            subcategory_idx = self.subcategory_dict.get(news_row['SubCategory'], 0)
            
            return {
                'user_idx': torch.tensor(user_idx, dtype=torch.long),
                'title_seq': torch.tensor(title_seq, dtype=torch.long),
                'abstract_seq': torch.tensor(abstract_seq, dtype=torch.long),
                'category_idx': torch.tensor(category_idx, dtype=torch.long),
                'subcategory_idx': torch.tensor(subcategory_idx, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float)
            }
        except KeyError:
            # Handle missing news (return zeros)
            return {
                'user_idx': torch.tensor(user_idx, dtype=torch.long),
                'title_seq': torch.zeros(self.max_title_len, dtype=torch.long),
                'abstract_seq': torch.zeros(self.max_abstract_len, dtype=torch.long),
                'category_idx': torch.tensor(0, dtype=torch.long),
                'subcategory_idx': torch.tensor(0, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float)
            }