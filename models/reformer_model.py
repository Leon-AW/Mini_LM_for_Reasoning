import torch
import torch.nn as nn
from transformers import ReformerConfig, ReformerModel
from gensim.models import Word2Vec

   # Load the trained Word2Vec model
   word2vec_model = Word2Vec.load("wikitext103_word2vec.model")

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