import torch
import torch.nn as nn

# Sample corpus
corpus = [
    "I am a boy",
    "I live in India",
    "In the city of Kolkata",
    "Kolkata is city of Joy",
    "It was the British capital of India",
    "Rosogolla is famous here",
    "We have Victoria Memorial , Indian Mueseun, Princep ghat",
    "Come visit us , we love people , we dont judge ",
]

# Step 1: Create a simple vocabulary mapping words to token IDs
vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}  # Special tokens
for sentence in corpus:
    for word in sentence.split():
        if word.lower() not in vocab:
            vocab[word.lower()] = len(vocab)

# Step 2: Convert sentences to token IDs
tokenized_corpus = [[vocab[word.lower()] for word in sentence.split()] for sentence in corpus]

# Step 3: Pad sequences to the same length
max_len = max(len(sentence) for sentence in tokenized_corpus)
padded_corpus = [sentence + [vocab["<PAD>"]] * (max_len - len(sentence)) for sentence in tokenized_corpus]

# Convert to tensor
input_tensor = torch.tensor(padded_corpus)

#print("Tokenized Corpus:\n", tokenized_corpus)
#print("Padded Tensor:\n", input_tensor)

# Expose required objects
__all__ = ["input_tensor", "vocab"]

