import numpy as np
import torch
import torch.nn as nn
from transformers import ReformerConfig, ReformerModel
import os
from gensim.models import Word2Vec

# Get the path to the embeddings directory
embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'embeddings')

# Load the Word2Vec model using the full path
word2vec_model = Word2Vec.load(os.path.join(embeddings_dir, "wikitext103_word2vec.model"))

# Manually load vectors if needed
vectors = np.load(os.path.join(embeddings_dir, "wikitext103_word2vec.model.wv.vectors.npy"), allow_pickle=True)

# Create an embedding matrix
vocab = word2vec_model.wv.key_to_index
embedding_dim = word2vec_model.vector_size
embedding_matrix = torch.zeros((len(vocab), embedding_dim))

for word, idx in vocab.items():
    embedding_matrix[idx] = torch.tensor(word2vec_model.wv[word])

# Define the model with the pre-trained embeddings
class ReformerWithCustomEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings):
        super(ReformerWithCustomEmbeddings, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        config = ReformerConfig(vocab_size=vocab_size)
        self.reformer = ReformerModel(config)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        outputs = self.reformer(inputs_embeds=embeddings)
        return outputs

# Initialize the model
model = ReformerWithCustomEmbeddings(len(vocab), embedding_dim, embedding_matrix)
