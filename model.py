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
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    
class FeedForwardBlock(nn.Module):
    '''
    first layer (linear_1) -> expands d_model (small dimension) to d_ff (larger dimension), this gives model more capacity to learn complex features
    ReLu Activation (torch.relu) -> introduces non-linearity to learn complex features
    self.dropout -> prevents over-fitting by randomly setting some values to zero
    linear_2 -> shrinks back from d_ff -> d_model ; to keep dimensions consistent
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # expands dimension    
        self.dropout = nn.Dropout(dropout) # Applies dropout
        self.linear_2 = nn.Linear(d_ff, d_model) # shrinks dimension back
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h
        # linear layers for query, key and value transformations
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model) # output projections
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention (query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # scaled dot-product attention
        # (Batch, h, seq_Len, d_k) --> (Batch, h, seq_Len, seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
        
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_Len, d_model) --> (Batch, seq_Len, d_model)
        key = self.w_k(k) # (Batch, seq_Len, d_model) --> (Batch, seq_Len, d_model)
        value = self.w_v(v) # (Batch, seq_Len, d_model) --> (Batch, seq_Len, d_model)
        
        # (Batch, seq_Len, d_model) --> (Batch, seq_Len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (Batch, h, seq_Len, d_k) --> (batch, seq_Len, h, d_k) --> (Batch, seq_Len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # (Batch, seq_Len, d_model) --> (Batch, seq_Len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
        
        
    
   