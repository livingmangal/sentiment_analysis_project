# import requests

# url = "http://127.0.0.1:5000/predict"
# data = {"text": "I love this product!"}
# response = requests.post(url, json=data)

# print(response.json())  # Expected: {'sentiment': ...}


vocab = {"i": 0, "love": 1, "this": 2, "movie": 3, "so": 4, "much": 5}
sentence = [0, 1, 2, 3]  # "I love this movie"
import torch
import torch.nn as nn

# Define parameters
vocab_size = len(vocab)  # Total words in vocab (6 words)
embedding_dim = 3        # Each word is represented as a 3D vector

# Create an embedding layer
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Convert sentence to a tensor
sentence_tensor = torch.tensor(sentence)  # Convert to PyTorch tensor

# Pass the sentence through the embedding layer
# embedded_sentence = embedding_layer(sentence_tensor)

# Print the output
print("Word Indices:", sentence_tensor)
# print("Embedded Vectors:\n", embedded_sentence)

