import torch
import torch.nn as nn
from MMHA import self_attention_output,MaskedSelfAttention  # Import output from MMHA

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)  # Expand to hidden_dim
        self.relu = nn.ReLU()  # Non-linearity
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)  # Project back to embedding_dim
        self.layer_norm = nn.LayerNorm(embedding_dim)  # Normalization
    
    def forward(self, x):
        residual = x  # Save original input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layer_norm(x + residual)  # Add residual connection and normalize
        return x

# Initialize FFN
embedding_dim = 16
hidden_dim = embedding_dim * 4  # Standard Transformer FFN size

ffn = FeedForwardNetwork(embedding_dim, hidden_dim)

# Apply FFN to self-attention output
ffn_output = ffn(self_attention_output)

# Expose required objects
__all__ = ["FeedForwardNetwork", "ffn_output"]

