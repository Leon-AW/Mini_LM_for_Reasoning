from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('./mini_lm')
tokenizer = GPT2Tokenizer.from_pretrained('./mini_lm')

# Set the pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Enter a prompt (or 'quit' to exit):")
while True:
    prompt = input("> ")
    if prompt.lower() == 'quit':
        break
    generated_text = generate_text(prompt)
    print(f"Generated text: {generated_text}\n")
