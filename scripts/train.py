import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from transformers import Trainer, TrainingArguments, ReformerTokenizerFast
from models.reformer_model import ReformerWithCustomEmbeddings
from utils.data_utils import load_dataset
import torch

def get_reformer_model():
    # Load the trained Word2Vec model
    from gensim.models import Word2Vec
    word2vec_model = Word2Vec.load("embeddings/wikitext103_word2vec.model")

    # Create an embedding matrix
    vocab = word2vec_model.wv.key_to_index
    embedding_dim = word2vec_model.vector_size
    embedding_matrix = torch.zeros((len(vocab), embedding_dim))

    for word, idx in vocab.items():
        embedding_matrix[idx] = torch.tensor(word2vec_model.wv[word])

    # Initialize the Reformer model with custom embeddings
    model = ReformerWithCustomEmbeddings(len(vocab), embedding_dim, embedding_matrix)
    return model

def train():
    # Check if GPU is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_reformer_model().to(device)  # Move model to GPU if available

    # Load a pre-trained fast tokenizer
    tokenizer = ReformerTokenizerFast.from_pretrained('google/reformer-enwik8')

    train_dataset, val_dataset = load_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=5000,
        logging_dir='./logs',
        logging_steps=500,
        logging_strategy="steps",  # Log at each step
        logging_first_step=True,   # Log the first step
        report_to="all",           # Report to all available integrations (e.g., console, TensorBoard)
        load_best_model_at_end=True,  # Load the best model at the end of training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    train()
