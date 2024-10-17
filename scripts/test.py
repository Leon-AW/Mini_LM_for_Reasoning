import torch
from transformers import ReformerTokenizerFast, ReformerForMaskedLM

def test_model(input_text):
    # Load the tokenizer and model
    tokenizer = ReformerTokenizerFast.from_pretrained("./reformer-wiki-100k")
    model = ReformerForMaskedLM.from_pretrained("./reformer-wiki-100k")
    model.eval()  # Set the model to evaluation mode

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")

    # Generate predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Decode the predictions
    predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
    return predicted_text

if __name__ == "__main__":
    while True:
        # Allow user to input custom text
        input_text = input("Enter your text (or type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break
        predicted_text = test_model(input_text)
        print(f"Input: {input_text}")
        print(f"Predicted: {predicted_text}")
