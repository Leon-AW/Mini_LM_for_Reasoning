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
    def __init__(self, tokenizer, file_path, block_size=128, max_size=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        if max_size:
            lines = lines[:max_size]
            logger.info(f"Limited dataset size to {max_size} samples")
        
        logger.info("Tokenizing data")
        self.examples = tokenizer(lines, truncation=True, max_length=block_size, padding="max_length")

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, i):
        # Return input_ids and labels (which are the same for masked language modeling)
        return {
            'input_ids': torch.tensor(self.examples['input_ids'][i]),
            'attention_mask': torch.tensor(self.examples['attention_mask'][i]),
            'labels': torch.tensor(self.examples['input_ids'][i])  # Use input_ids as labels for MLM
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing tokenizer and model")
    tokenizer = ReformerTokenizerFast.from_pretrained("google/reformer-crime-and-punishment")
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    config = ReformerConfig.from_pretrained("google/reformer-crime-and-punishment")
    config.is_decoder = False
    
    # **WICHTIG:** Anpassen der axial_pos_shape
    config.axial_pos_shape = (16, 8)  # 16 * 8 = 128, passend zur block_size

    model = ReformerForMaskedLM.from_pretrained(
        "google/reformer-crime-and-punishment", 
        config=config,
        ignore_mismatched_sizes=True  # Add this parameter
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Loading and splitting dataset")
    full_dataset = WikiDataset(tokenizer, "data/raw/wiki.train.tokens", max_size=1000)  # Weiter reduzierte Größe
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    logger.info("Defining training arguments")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Weiter reduzierte Batch-Größe
        per_device_eval_batch_size=2,   
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,  # Beibehaltung der Gradientenakkumulation
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