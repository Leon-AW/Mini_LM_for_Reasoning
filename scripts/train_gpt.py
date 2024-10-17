# Import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
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

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Preprocess your data
def preprocess_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    tokenized_data = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    print("Tokenized data structure:", tokenized_data)  # Debugging print statement
    return TextDataset(tokenized_data)

# Load and preprocess the dataset
train_dataset = preprocess_data('data/raw/wiki.train.tokens')

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

# Define a simple data collator
def data_collator(features):
    return {key: torch.stack([f[key] for f in features]) for key in features[0].keys()}

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
