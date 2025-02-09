import torch
import torch.nn as nn
import torch.optim as optim
from Decoder import TransformerDecoder

# Define hyperparameters
vocab_size = 10000  # Adjust based on your dataset
embedding_dim = 16
num_heads = 4
hidden_dim = embedding_dim * 4
num_layers = 6
learning_rate = 1e-4

# Initialize model
model = TransformerDecoder(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers)

# Loss function (CrossEntropyLoss for language modeling)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignoring padding token

# Optimizer (Adam is commonly used in transformers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Expose required objects
__all__ = ["model", "criterion", "optimizer"]
