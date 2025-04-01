import torch
import torch.nn as nn
import math

'''
This class maps input tokens (wprds) into 
dense vector representations (embeddings)

'''
class InputEmbeddings(nn.Module):
    # d_model is the size of each embedding vector
    # vocab_size is the total number of unique tokens in the vocabulary
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__() # calls the parent class nn.module constructor
        self.d_model = d_model
        self.vocab_size = vocab_size
        # creates an embedding layer that learns a d_model-dimensional vector representation of each token in vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    # defines the forward pass
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # retieves the embeddings for the input tokens x and multiplies the embeddings to scale them to maintain stable gradients

'''
This class adds positional encoding to token embeddings so the model understands order
'''
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # dimension of embeddings
        self.seq_len = seq_len # the maximum sequence length
        self.dropout = nn.Dropout(dropout) # the dropout rate to regularize embeddings
        
        pe = torch.zeros(seq_len, d_model)
        
        # creates a tensor of positions [0,1,2, ...., seq_len-1]
        # reshapes it to a column vector (unsqueeze(1) makes it of shape (seq__len,1))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # computes the scaling term for positional encodings
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(1000.0)/d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # applies sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term) # applies cosine to odd indices
        
        pe = pe.unsqueeze(0) # adds a batch dimension
        
        self.register_buffer('pe', pe) # stores pe as buffer so it won't be updated during training but remain part of the model
    
    # this function adds positional encoding to the input
    def forward(self, x):
        # Adds positional encoding to x
        # .requires_grad_(False) ensures that pe is not updated during backpropagation
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x) # applies dropout for regularization
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha =  nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # Additive
        
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True) # averages values per feature (across the last dimension)
        std = x.std(dim = -1, keepdim = True) # measures how spread out values are
        # subtract the mean -> centers values around 0
        # divide by std -> makes variance 1
        # add eps to prevent division by zero
        # alpha learns how much to scale (like batch norm's gamma)
        # bias learns how much to shift (like batch norm's beta)
        return self.alpha *(x - mean) / (std + self.eps) + self.bias
    
   