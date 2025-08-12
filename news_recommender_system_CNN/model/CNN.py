import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsRecommendationCNN(nn.Module):
    def __init__(self, embedding_matrix, num_users, num_categories, num_subcategories, 
                 embed_dim=100, num_filters=128, filter_sizes=[3, 4, 5], 
                 user_embed_dim=50, category_embed_dim=20, dropout=0.3):
        super(NewsRecommendationCNN, self).__init__()
        
        vocab_size = embedding_matrix.shape[0]
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        
        # Word embeddings (pre-trained GloVe)
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.word_embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))
        self.word_embedding.weight.requires_grad = True  # Fine-tune embeddings
        
        # User embeddings
        self.user_embedding = nn.Embedding(num_users + 1, user_embed_dim)
        
        # Category embeddings
        self.category_embedding = nn.Embedding(num_categories + 1, category_embed_dim)
        self.subcategory_embedding = nn.Embedding(num_subcategories + 1, category_embed_dim)
        
        # CNN layers for title
        self.title_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in filter_sizes
        ])
        
        # CNN layers for abstract
        self.abstract_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Calculate final feature dimension
        cnn_output_dim = len(filter_sizes) * num_filters * 2  # title + abstract
        total_dim = cnn_output_dim + user_embed_dim + category_embed_dim * 2
        
        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_idx, title_seq, abstract_seq, category_idx, subcategory_idx):
        batch_size = title_seq.size(0)
        
        # User embeddings
        user_emb = self.user_embedding(user_idx)
        
        # Category embeddings
        cat_emb = self.category_embedding(category_idx)
        subcat_emb = self.subcategory_embedding(subcategory_idx)
        
        # Word embeddings for title and abstract
        title_emb = self.word_embedding(title_seq)  # (batch, seq_len, embed_dim)
        abstract_emb = self.word_embedding(abstract_seq)
        
        # Transpose for CNN (batch, embed_dim, seq_len)
        title_emb = title_emb.transpose(1, 2)
        abstract_emb = abstract_emb.transpose(1, 2)
        
        # Apply CNN to title
        title_conv_outputs = []
        for conv in self.title_convs:
            conv_out = F.relu(conv(title_emb))  # (batch, num_filters, new_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
            title_conv_outputs.append(pooled)
        title_features = torch.cat(title_conv_outputs, dim=1)
        
        # Apply CNN to abstract
        abstract_conv_outputs = []
        for conv in self.abstract_convs:
            conv_out = F.relu(conv(abstract_emb))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            abstract_conv_outputs.append(pooled)
        abstract_features = torch.cat(abstract_conv_outputs, dim=1)
        
        # Combine all features
        combined_features = torch.cat([
            user_emb, title_features, abstract_features, cat_emb, subcat_emb
        ], dim=1)
        
        # Apply dropout
        combined_features = self.dropout(combined_features)
        
        # Final prediction
        output = self.fc(combined_features)
        
        return output.squeeze(1)