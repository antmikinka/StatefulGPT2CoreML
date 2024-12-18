import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import coremltools as ct

class SimpleAttentionWithKeyValueCache(nn.Module):
	"""Add kv-cache into SimpleAttention."""

	def __init__(self, embed_size):
		super().__init__()
		self.query = nn.Linear(embed_size, embed_size)
		self.key = nn.Linear(embed_size, embed_size)
		self.value = nn.Linear(embed_size, embed_size)

	def forward(self, x, attention_mask, k_cache, v_cache):
		Q = self.query(x)
		newly_computed_k = self.key(x)
		newly_computed_v = self.value(x)

		# Update kv-cache in-place.
		q_len = Q.shape[-2]
		end_step = attention_mask.shape[-1]
		past_kv_len = end_step - q_len
		k_cache[:, past_kv_len:end_step, :] = newly_computed_k
		v_cache[:, past_kv_len:end_step, :] = newly_computed_v

		# The K and V we need is (batch_size, q_len + past_kv_len, embed_size).
		K = k_cache[:, :end_step, :]
		V = v_cache[:, :end_step, :]

		return torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask)

class GPT2WithKVCache(nn.Module):
	def __init__(self, vocab_size, embed_size, batch_size, max_seq_len):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.attention = SimpleAttentionWithKeyValueCache(embed_size)
		self.fc = nn.Linear(embed_size, embed_size)

		# Initialize KV cache
		self.kvcache_shape = (batch_size, max_seq_len, embed_size)
		self.register_buffer("k_cache", torch.zeros(self.kvcache_shape))
		self.register_buffer("v_cache", torch.zeros(self.kvcache_shape))

	def forward(self, input_ids, causal_mask):
		embedded = self.embedding(input_ids)
		attention_output = self.attention(embedded, causal_mask, self.k_cache, self.v_cache)
		return self.fc(attention_output)

def download_and_prepare_gpt2_model(vocab_size=50257, embed_size=768, batch_size=1, max_seq_len=1024):
	print("Downloading GPT-2 model from Hugging Face...")
	# Load the pre-trained Hugging Face GPT-2 model
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
	gpt2.eval()

	print("Rebuilding GPT-2 model with KV-cache support...")
	# Recreate the GPT-2 model with Key-Value Cache
	model_with_kvcache = GPT2WithKVCache(vocab_size=vocab_size, embed_size=embed_size, batch_size=batch_size, max_seq_len=max_seq_len)
	model_with_kvcache.load_state_dict(gpt2.state_dict(), strict=False)
	model_with_kvcache.eval()

	return model_with_kvcache, tokenizer

def convert_to_coreml(model_with_kvcache, max_seq_len=1024):
	print("Converting the model to Core ML...")

	# Prepare the input
	batch_size = 1
	seq_len = 5
	vocab_size = 50257
	embed_size = 768
	input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
	causal_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32)

	# Trace the PyTorch model
	traced_model = torch.jit.trace(model_with_kvcache, [input_ids, causal_mask])

	# Define Core ML model inputs, outputs, and states (KV cache)
	query_length = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
	end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
	inputs = [
		ct.TensorType(shape=(batch_size, query_length), dtype=np.int32, name="input_ids"),
		ct.TensorType(shape=(batch_size, query_length, end_step_dim), dtype=np.float16, name="causal_mask"),
	]
	outputs = [ct.TensorType(dtype=np.float16, name="output")]

	# KV-cache states
	states = [
		ct.StateType(
			wrapped_type=ct.TensorType(
				shape=model_with_kvcache.kvcache_shape, dtype=np.float16
			),
			name="k_cache",
		),
		ct.StateType(
			wrapped_type=ct.TensorType(
				shape=model_with_kvcache.kvcache_shape, dtype=np.float16
			),
			name="v_cache",
		),
	]

	# Convert to Core ML
	converted_model = ct.convert(
		traced_model,
		inputs=inputs,
		outputs=outputs,
		states=states,
		minimum_deployment_target=ct.target.iOS18,
		compute_units=ct.ComputeUnit.CPU_AND_GPU,
	)

	# Save the converted model
	model_path = "gpt2_with_kvcache_try2.mlpackage"
	converted_model.save(model_path)
	print(f"Model successfully converted and saved to {model_path}")

def run_pytorch_prediction(model, tokenizer, sentence_fragment):
	print("Running PyTorch prediction...")
	context = torch.tensor(tokenizer.encode(sentence_fragment)).unsqueeze(0)
	torch_out = model(context, torch.zeros((1, context.size(1), context.size(1)), dtype=torch.float32))
	generated_text_torch = tokenizer.decode(torch_out.argmax(-1).flatten())
	print(f"Completed: {generated_text_torch}")

def run_coreml_prediction(mlmodel_path, tokenizer, sentence_fragment):
	print("Running Core ML prediction...")
	# Load the Core ML model
	mlmodel = ct.models.MLModel(mlmodel_path)
	
	# Prepare input
	context = torch.tensor(tokenizer.encode(sentence_fragment)).unsqueeze(0)
	causal_mask = torch.zeros((1, context.size(1), context.size(1)), dtype=torch.float32)  # Causal mask
	
	# Initialize the state for KV cache
	kv_cache_state = mlmodel.make_state()  # Create the state for kv_cache (k_cache, v_cache)
	
	# Convert inputs to the appropriate format for Core ML
	coreml_inputs = {
		"input_ids": context.to(torch.int32).numpy(),
		"causal_mask": causal_mask.numpy().astype(np.float16)  # Add the causal mask
	}
	
	# Perform prediction with the initialized state
	prediction_dict = mlmodel.predict(coreml_inputs, state=kv_cache_state)
	
	# Get the output from prediction (limiting output size for debugging)
	generated_tensor = prediction_dict["output"]
	
	# DEBUG: Print raw output tensor for inspection
	print("Raw Output Tensor:", generated_tensor)
	
	# Limit the length of the generated sequence to avoid overly long output
	max_length = 50  # Adjust this to a reasonable length
	limited_output = generated_tensor.flatten()[:max_length]
	
	# Decode and print the generated text
	generated_text = tokenizer.decode(limited_output)
	print(f"Completed: {generated_text}")

def main():
	# Step 1: Download and prepare GPT-2 model with KV-cache support
	model_with_kvcache, tokenizer = download_and_prepare_gpt2_model()

	# Step 2: Convert the PyTorch model to Core ML
	convert_to_coreml(model_with_kvcache)

	# Step 3: Run prediction with PyTorch
	sentence_fragment = "The quick brown fox"
	run_pytorch_prediction(model_with_kvcache, tokenizer, sentence_fragment)

	# Step 4: Run prediction with the Core ML model
	mlmodel_path = "gpt2_with_kvcache_try2.mlpackage"
	run_coreml_prediction(mlmodel_path, tokenizer, sentence_fragment)

if __name__ == "__main__":
	main()