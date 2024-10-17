# Import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Preprocess your data
def preprocess_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    tokenized_data = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    print("Tokenized data structure:", tokenized_data)  # Debugging print statement
    return tokenized_data

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
    print("Features received by data collator:", features)  # Debugging print statement
    # Ensure that each feature is a dictionary with 'input_ids' and 'attention_mask'
    try:
        input_ids = torch.stack([f['input_ids'].squeeze() for f in features])
        attention_mask = torch.stack([f['attention_mask'].squeeze() for f in features])
        labels = input_ids.clone()  # Clone input_ids for labels
    except KeyError as e:
        print(f"KeyError in data collator: {e}")
        print("Feature keys:", [f.keys() for f in features])
        raise

    print("Data collator output:", {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels})  # Debugging print statement
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

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
