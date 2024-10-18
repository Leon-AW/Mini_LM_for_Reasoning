# Import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up cache directory
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

# Load tokenizer and model with explicit cache directory
try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir).to(device)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Attempting to download manually...")
    # If loading fails, try to manually download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=cache_dir)

# Increase the model's maximum sequence length
model.config.max_position_embeddings = 2048  # or any larger value you need
model.resize_token_embeddings(len(tokenizer))

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        self.examples = []
        for line in lines:
            tokenized_text = tokenizer.encode(line)
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenized_text[i:i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

# Load and preprocess the dataset
train_dataset = TextDataset(tokenizer, 'data/raw/wiki.train.tokens', block_size=1024)

print(f"Number of training examples: {len(train_dataset)}")

# Define training arguments with TensorBoard logging
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size
    gradient_accumulation_steps=4,  # Accumulate gradients
    fp16=True,  # Enable mixed precision training
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=500,  # Log every 500 steps
)

# Use DataCollatorForLanguageModeling for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./mini_lm')
tokenizer.save_pretrained('./mini_lm')
