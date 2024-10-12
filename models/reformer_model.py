import torch
import torch.nn as nn
from transformers import ReformerConfig, ReformerModel, ReformerForMaskedLM
import os
from gensim.models import Word2Vec

# Get the path to the embeddings directory
embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'embeddings')

# Load the Word2Vec model using the full path
word2vec_model = Word2Vec.load(os.path.join(embeddings_dir, "wikitext103_word2vec.model"))

# Create an embedding matrix
vocab = word2vec_model.wv.key_to_index
embedding_dim = word2vec_model.vector_size
embedding_matrix = torch.zeros((len(vocab), embedding_dim))

for word, idx in vocab.items():
    embedding_matrix[idx] = torch.tensor(word2vec_model.wv[word])

# Define the model with the pre-trained embeddings
class ReformerWithCustomEmbeddings(nn.Module):
    def __init__(self, config, pretrained_embeddings):
        super(ReformerWithCustomEmbeddings, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.reformer = ReformerForMaskedLM(config)
        
        # Replace the reformer's word embeddings with our custom embeddings
        self.reformer.set_input_embeddings(self.embedding)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.reformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
