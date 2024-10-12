import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ReformerTokenizerFast, ReformerForMaskedLM, ReformerConfig, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikiDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        logger.info("Tokenizing data")
        self.examples = tokenizer(lines, truncation=True, max_length=block_size, padding="max_length")

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, i):
        return {key: torch.tensor(val[i]) for key, val in self.examples.items()}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing tokenizer and model")
    tokenizer = ReformerTokenizerFast.from_pretrained("google/reformer-crime-and-punishment")
    
    # Konfiguration für das Masked Language Model sicherstellen
    config = ReformerConfig.from_pretrained("google/reformer-crime-and-punishment")
    config.is_decoder = False  # Sicherstellen, dass is_decoder=False gesetzt ist
    
    model = ReformerForMaskedLM.from_pretrained("google/reformer-crime-and-punishment", config=config).to(device)

    logger.info("Loading and splitting dataset")
    full_dataset = WikiDataset(tokenizer, "data/raw/wiki.train.tokens")
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    logger.info("Defining training arguments")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model and tokenizer")
    model.save_pretrained("./reformer-wiki")
    tokenizer.save_pretrained("./reformer-wiki")

if __name__ == "__main__":
    main()