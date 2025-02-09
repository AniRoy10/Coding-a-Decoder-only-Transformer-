import torch
import torch.nn as nn
from tokens import input_tensor, vocab  # Import tokenized data

# Define embedding size
embedding_dim = 16  # Each word will be represented by a 16-dimensional vector

# Step 4: Create an embedding layer
embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)

# Convert token IDs into embeddings
embedded_input = embedding_layer(input_tensor)

# Expose required objects
__all__ = ["embedded_input", "embedding_layer"]

