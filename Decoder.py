import torch
import torch.nn as nn
from Transformer import TransformerBlock

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len=1000):
        super(TransformerDecoder, self).__init__()

        # Token Embedding + Positional Encoding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)

        # Transformer Blocks (Stacked)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])

        # Final Linear layer to map embeddings back to vocab size
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Token and Positional Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device).unsqueeze(0))

        x = token_emb + pos_emb  # Combine token + positional embeddings

        # Pass through Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final Linear layer for output probabilities
        logits = self.fc_out(x)

        return logits

# Example usage
if __name__ == "__main__":
    vocab_size = 10000
    embedding_dim = 16
    num_heads = 4
    hidden_dim = embedding_dim * 4
    num_layers = 6  # Number of stacked Transformer blocks

    decoder = TransformerDecoder(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers)

    # Example input: batch_size=2, seq_len=5 (random token indices)
    sample_input = torch.randint(0, vocab_size, (2, 5))
    output = decoder(sample_input)

    print("Decoder Output Shape:", output.shape)  # Expected: (batch_size, seq_len, vocab_size)
