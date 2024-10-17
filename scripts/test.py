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

        # Check if the mask token is in the input
        if tokenizer.mask_token_id not in inputs["input_ids"]:
            print("Error: The input must contain a [MASK] token.")
            continue

        # Generate predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted token IDs for the masked position
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        predicted_token_ids = logits[0, mask_token_index, :].argmax(dim=-1)

        # Decode the predicted tokens to text
        predicted_tokens = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)

        print(f"Predicted: {predicted_tokens}")

# Run the test model function
test_model()
