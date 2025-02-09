Overview

This repository implements a Decoder-Only Transformer from scratch using Python and PyTorch. The goal is to build a transformer model that can generate text based on an input prompt by predicting the next token in a sequence. We break down the implementation into key building blocks and train the model step by step.

Architecture

A Decoder-Only Transformer consists of multiple decoder layers stacked on top of each other. Each layer contains:

Token Embeddings

Positional Encodings

Self-Attention Mechanism (Masked)

Feedforward Neural Network

Layer Normalization & Residual Connections

Unlike a full transformer (which includes an encoder), a decoder-only model generates output token-by-token and uses masked attention to prevent looking ahead.

Main Building Blocks

1. Tokenization (tokens.py)

The first step is tokenizing the input corpus, meaning we assign token IDs to each word.

✅ What happens here?

A vocabulary is built from the dataset.

Each word is mapped to a unique integer (token ID).

Unknown words are handled with a special <UNK> token.

Example:

vocab = {"I": 1, "love": 2, "Transformers": 3, "<PAD>": 0}
sentence = "I love Transformers"
tokens = [1, 2, 3]

2. Embeddings (embeddings.py)

Since token IDs are just numbers, we convert them into dense vectors called embeddings. These embeddings capture semantic relationships between words.

✅ How it works?

Each token ID is mapped to a trainable embedding vector.

Example:

token_id = 1  # "I"
embedding_vector = embedding_matrix[token_id]  # Get the learned vector for "I"

These embeddings are learned during training.

3. Positional Encoding (positional_encoding.py)

Transformers do not have recurrence (RNNs) or convolution (CNNs), so we need to explicitly encode the order of words in a sentence using positional encodings.

✅ What happens here?

Each token gets a unique positional vector added to its embedding.

Uses sinusoidal functions to create patterns.

Example:

pos_encoding = sin(position / 10000^(2i/dim))

Helps the model understand sequence order.

4. Masked Self-Attention (attention.py)

Self-attention allows the model to focus on different words in the sentence while processing a token.

✅ Key Steps:

Compute Queries, Keys, and Values

Scale Dot-Product Attention:

attention_scores = softmax((Q @ K.T) / sqrt(d_k))

Mask Future Tokens: Prevents the model from seeing future words.

Multiply by Values and Sum Up

✅ Example:

If input is "I love Transformers", at position t, the model can only attend to words before or at t, not after.

5. Feedforward Network (ffn.py)

Each decoder layer contains a fully connected feedforward network.

✅ Steps:

First linear layer transforms embedding space.

Apply an activation function (e.g., ReLU, GELU).

Second linear layer projects back to original dimension.

6. Transformer Decoder Block (decoder.py)

Each decoder block consists of:

Masked Self-Attention

Feedforward Network

Layer Normalization

Residual Connections

✅ Example Flow:

Input sentence is tokenized and embedded.

Positional encodings are added.

Data flows through multiple stacked decoder layers.

Output token probabilities are generated.

7. Dataset Loader (dataset_loader.py)

We prepare input-target pairs for training.

✅ How it works?

Takes a dataset and converts sentences into fixed-length sequences.

Shifts the input to create a target sequence.

✅ Example:
Original: "I love Transformers"

Input:   [1, 2, 3, 0, 0]  # Padded sequence
Target:  [2, 3, 0, 0, 0]  # Next-word prediction

8. Training (train.py)

This script trains the transformer using the dataset.

✅ Key Steps:

Forward Pass: Input is passed through the transformer.

Loss Computation: Compare predicted and actual tokens.

Backpropagation & Optimization: Update weights.

Repeat until convergence.

✅ Loss Function:
We use Cross-Entropy Loss because we are predicting a probability distribution over vocabulary tokens.

loss = CrossEntropyLoss(predicted_tokens, target_tokens)

✅ Optimizer:
We use Adam with learning rate warm-up and decay.

optimizer = AdamW(model.parameters(), lr=learning_rate)

9. Inference (generate.py)

After training, we use the model for text generation.

✅ Steps:

Provide an input prompt.

Generate tokens one by one until reaching <EOS>.

Convert token IDs back to words.

✅ Example:

prompt = "I love"
generated = model.generate(prompt, max_length=20)
print(generated)

🔹 Output: "I love Transformers and AI models!"

Putting It All Together

✅ Step 1: Preprocess dataset (tokenization, embedding).✅ Step 2: Define transformer model (decoder layers, attention, FFN).✅ Step 3: Train the model (loss function, optimizer).✅ Step 4: Generate text with trained model.

Conclusion

This project demonstrates how to build a Decoder-Only Transformer from scratch. We covered:

Tokenization, embeddings, and positional encodings.

Self-attention and masked decoding.

Training and inference pipeline.

This model serves as a foundation for auto-regressive text generation tasks, such as:

Chatbots
Code generation
AI writing assistants 
