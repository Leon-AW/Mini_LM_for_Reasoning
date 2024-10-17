# Import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Preprocess your data
def preprocess_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# Load and preprocess the dataset
train_data = preprocess_data('data/raw/wiki.train.tokens')

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
    return {'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['input_ids'] for f in features])}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./mini_lm')
tokenizer.save_pretrained('./mini_lm')
