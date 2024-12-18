import numpy as np
import coremltools as ct
from transformers import GPT2Tokenizer
import os

# Function to load Core ML model
def load_model():
	print("Loading Core ML model...")
	model_path = "gpt2_with_kvcache.mlpackage"  # Make sure this is the path to your Core ML model
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Core ML model not found at {model_path}")
	
	# Load the CoreML model
	model = ct.models.MLModel(model_path)
	return model

# Function to load GPT-2 tokenizer
def load_tokenizer():
	print("Loading GPT-2 tokenizer...")
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	return tokenizer

# Function to perform prediction using the Core ML model
def predict_with_coreml(model, input_text, tokenizer, kv_cache_state):
	# Set pad_token to eos_token if not already set
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Tokenize input text
	input_tokens = tokenizer(input_text, return_tensors="np", truncation=True, padding=True)
	
	input_ids = input_tokens["input_ids"]
	print(f"Tokenized input: {input_ids}")

	# Create the necessary inputs for CoreML
	inputs = {
		"input_ids": input_ids,
		"full_sequence_length": np.array([input_ids.shape[1]], dtype=np.int32)
	}

	# Perform prediction
	outputs = model.predict(inputs, state=kv_cache_state)

	# Get the logits and kv_cache from the model
	logits = outputs["logits"]
	kv_cache_state = outputs.get("kv_cache_state")

	# Get the predicted token ids from the logits
	predicted_token_ids = np.argmax(logits, axis=-1)  # Get the predicted token ids
	predicted_text = tokenizer.decode(predicted_token_ids.flatten())  # Decode token ids to text

	return predicted_text, kv_cache_state

# Main function to run the model and generate predictions
def main():
	model = load_model()  # Load your CoreML model
	tokenizer = load_tokenizer()  # Load GPT-2 tokenizer from HuggingFace
	
	input_text = input("Enter text for prediction: ")  # Get input from the user
	print(f"Generating prediction for: '{input_text}'")

	kv_cache_state = None  # Initialize kv_cache_state as None for the first prediction

	# Run the prediction and get the full generated text
	predicted_text, kv_cache_state = predict_with_coreml(model, input_text, tokenizer, kv_cache_state)

	# Output the generated text
	print(f"Generated text: {predicted_text}")

if __name__ == "__main__":
	main()