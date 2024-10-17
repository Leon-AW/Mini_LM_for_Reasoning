from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('./mini_lm')
tokenizer = GPT2Tokenizer.from_pretrained('./mini_lm')

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Enter a prompt (or 'quit' to exit):")
while True:
    prompt = input("> ")
    if prompt.lower() == 'quit':
        break
    generated_text = generate_text(prompt)
    print(f"Generated text: {generated_text}\n")