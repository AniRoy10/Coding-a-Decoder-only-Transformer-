import torch
import torch.nn as nn
import math
from positional_encoding import embedded_input_with_pos  # Import encoded input

class MaskedSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MaskedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads  # Dimension per head
        
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        
        # Output linear transformation
        self.W_out = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create a mask to prevent attending to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores.masked_fill_(mask, float('-inf'))  # Mask future tokens

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to V
        attention_output = torch.matmul(attention_weights, V)

        # Reshape back to original shape (batch_size, seq_len, embedding_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Apply final linear layer
        return self.W_out(attention_output)

# Example usage
embedding_dim = 16
num_heads = 4
self_attention = MaskedSelfAttention(embedding_dim, num_heads)

# Apply masked self-attention to encoded input
self_attention_output = self_attention(embedded_input_with_pos)

# Expose required objects
__all__ = ["MaskedSelfAttention", "self_attention_output"]

