# Step 1: Load the corpus from a text file

def load_corpus(file_path="corpus.txt"):
    """Reads the corpus from a file and returns a list of sentences."""
    with open(file_path, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f.readlines() if line.strip()]
    return corpus

# Step 2: Tokenize the corpus

def tokenize_corpus(corpus):
    """Tokenizes the given corpus into words and assigns unique indices."""
    token_to_id = {}
    id_to_token = {}
    current_id = 0
    
    for sentence in corpus:
        words = sentence.split()
        for word in words:
            if word not in token_to_id:
                token_to_id[word] = current_id
                id_to_token[current_id] = word
                current_id += 1
    
    return token_to_id, id_to_token

# Step 3: Load and tokenize
if __name__ == "__main__":
    corpus = load_corpus()
    token_to_id, id_to_token = tokenize_corpus(corpus)
    
    print("Token to ID:", token_to_id)
    print("ID to Token:", id_to_token)
