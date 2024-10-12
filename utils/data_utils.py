from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Use the encode method of SimpleTokenizer
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids)  # Assuming all tokens are attended to
        }

def load_dataset(tokenizer, max_length=128):
    # Load your actual data here
    train_texts = ["Your training sentence 1", "Your training sentence 2"]  # Replace with actual data
    val_texts = ["Your validation sentence 1", "Your validation sentence 2"]  # Replace with actual data

    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)

    return train_dataset, val_dataset
