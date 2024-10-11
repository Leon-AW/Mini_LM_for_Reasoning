from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training progress."""
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch} start")

    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch} end")
        self.epoch += 1

def train_word_embeddings(corpus_path, model_path, batch_size=1000):
    # Initialize the callback
    epoch_logger = EpochLogger()

    # Initialize Word2Vec model
    word2vec_model = Word2Vec(
        vector_size=300,
        window=5,
        min_count=1,
        workers=4,
        epochs=5,  # You can adjust the number of epochs
        callbacks=[epoch_logger]  # Add the callback here
    )

    # Build initial vocabulary
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        tokenized_corpus = [simple_preprocess(line) for line in lines[:batch_size]]  # Use the first batch to build initial vocab
        word2vec_model.build_vocab(tokenized_corpus)

    # Optionally update vocabulary with more data
    for i in tqdm(range(batch_size, len(lines), batch_size), desc="Updating Vocabulary"):
        batch = lines[i:i + batch_size]
        tokenized_batch = [simple_preprocess(line) for line in batch]
        word2vec_model.build_vocab(tokenized_batch, update=True)

    # Train the model in chunks
    for epoch in range(word2vec_model.epochs):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in tqdm(range(0, len(lines), batch_size), desc=f"Training Epoch {epoch+1}"):
                batch = lines[i:i + batch_size]
                tokenized_batch = [simple_preprocess(line) for line in batch]
                word2vec_model.train(tokenized_batch, total_examples=len(tokenized_batch), epochs=1)

    word2vec_model.save(model_path)

if __name__ == "__main__":
    train_word_embeddings('data/raw/wiki.train.tokens', 'embeddings/wikitext103_word2vec.model')