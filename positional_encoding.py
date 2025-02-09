import torch
import torch.nn as nn
import math

try:
    from embeddings import embedded_input  # Ensure `embedded_input` is available
except ImportError:
    print("Error: Unable to import 'embedded_input' from embeddings.py.")
    embedded_input = None

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        positional_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        if x is None:
            raise ValueError("Error: Input to PositionalEncoding is None.")
        return x + self.positional_encoding[:x.shape[1], :]

embedding_dim = 16
pos_encoder = PositionalEncoding(embedding_dim)

# Ensure embedded_input exists before applying positional encoding
if embedded_input is not None:
    embedded_input_with_pos = pos_encoder(embedded_input)
else:
    print("Warning: 'embedded_input' is None, cannot compute 'embedded_input_with_pos'.")
    embedded_input_with_pos = None

# Ensure `embedded_input_with_pos` is defined for import
__all__ = ["PositionalEncoding", "embedded_input_with_pos"]
