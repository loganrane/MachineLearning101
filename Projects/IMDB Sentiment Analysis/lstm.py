# create lstm.py

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        """
        :param embedding_matrix: numpy array with vectors for all words.
        """
        super(LSTM, self).__init__()
        # Number of words = Number of rows in embedding matrix
        num_words = embedding_matrix.shape[0]
        
        # Dimension of embedding is num of columns in the matrix
        embed_dim = embedding_matrix.shape[1]
        
        # We define the input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = num_words,
            embedding_dim = embed_dim
        )
        
        # Embedding matrix is used as weights of the embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype = torch.float32
            )
        )
        
        # We don't want to train the pretrained embeddings
        self.embedding.weight.requires_grad = False
        
        # A simple bidirectional LSTM with hidden size 128
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional = True,
            batch_first = True,
        )
        
        # Output layer which is a linear layer
        self.out = nn.Linear(512, 1)
        
        
    def forward(self, x):
        # Pass the data through embedding layer, the input is just tokens
        x = self.embedding(x)
        
        # Move the embedding ouput to lstm
        x, _ = self.lstm(x)
        
        # Apply mean and max pooling on lastm output
        avg_pool = torch.mean(x, 1)
        max_pool = torch.max(x, 1)
        
        # concatenate mean and max pooling
        # this is why size is 512
        # 128 for each direction = 256
        # avg_pool = 256 and max_pool = 256
        out = torch.cat((avg_pool, max_pool), 1)
        
        # pass through the output layer and return the output
        out = self.out(out)
        
        # return linear output
        return out