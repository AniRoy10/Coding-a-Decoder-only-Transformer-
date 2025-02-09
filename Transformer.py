import torch
import torch.nn as nn
from MMHA import MaskedSelfAttention
from FFNN import FeedForwardNetwork

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()

        # Multi-Head Self-Attention
        self.self_attention = MaskedSelfAttention(embedding_dim, num_heads)

        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(embedding_dim, hidden_dim)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Step 1: Multi-Head Self-Attention with Add & Norm
        attention_output = self.self_attention(x)
        x = self.norm1(x + attention_output)  # Add & Norm

        # Step 2: Feed-Forward Network with Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Add & Norm

        return x

# Example usage
if __name__ == "__main__":
    embedding_dim = 16
    num_heads = 4
    hidden_dim = embedding_dim * 4  # Standard FFN size in Transformers

    transformer_block = TransformerBlock(embedding_dim, num_heads, hidden_dim)

    # Sample input tensor (batch_size=2, seq_len=5, embedding_dim=16)
    sample_input = torch.randn(2, 5, embedding_dim)
    output = transformer_block(sample_input)

    print("Transformer Block Output Shape:", output.shape)
