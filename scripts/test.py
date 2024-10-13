import torch
from transformers import ReformerTokenizerFast, ReformerForMaskedLM

def test_model():
    # Load the tokenizer and model
    tokenizer = ReformerTokenizerFast.from_pretrained("./reformer-wiki")
    model = ReformerForMaskedLM.from_pretrained("./reformer-wiki")
    model.eval()  # Set the model to evaluation mode

    while True:
        # Get user input
        input_text = input("Enter a sentence with a [MASK] token (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted token IDs
        predicted_token_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted tokens to text
        predicted_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

        print(f"Predicted: {predicted_text}")

# Run the test model function
test_model()