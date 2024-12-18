import torch
import coremltools as ct
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# Define the Stateful model class using AutoModelForCausalLM
class StatefulModel(torch.nn.Module):
	def __init__(self, model_path, batch_size=1, context_size=128):
		super().__init__()
		# Load the model configuration to get n_layers, n_heads, and d_model dynamically
		# Load the model from Hugging Face using AutoModelForCausalLM
		self.model = AutoModelForCausalLM.from_pretrained(model_path)
		config = AutoConfig.from_pretrained(model_path)

		# Extract parameters dynamically
		n_layers = config.n_layer  # Number of layers
		n_heads = config.n_head    # Number of attention heads
		d_model = config.hidden_size  # Hidden dimension size

		
		# Initialize kv-cache placeholder (past_key_values should be a list of tuples of (key, value) for each layer)
		self.kv_cache_shape = (batch_size, n_heads, context_size, d_model // n_heads)
		
		# Initialize past_key_values cache with correct shape
		self.past_key_values = self._init_past_key_values(n_layers)

	def _init_past_key_values(self, n_layers):
		""" Initialize the past key-values cache with the correct shape. """
		return [(torch.zeros(self.kv_cache_shape, dtype=torch.float16),
				 torch.zeros(self.kv_cache_shape, dtype=torch.float16)) for _ in range(n_layers)]

	def forward(self, input_ids, attention_mask=None):
		# Check if the input sequence length has changed, and reset past_key_values if necessary
		current_seq_len = input_ids.shape[1]
		expected_seq_len = self.past_key_values[0][0].shape[2]
		if current_seq_len != expected_seq_len:
			# Reset past_key_values to match the new sequence length
			print(f"Resetting kv-cache: input sequence length changed from {expected_seq_len} to {current_seq_len}")
			self.past_key_values = self._init_past_key_values(len(self.past_key_values))

		# Perform forward pass with key-value caching
		outputs = self.model(input_ids=input_ids, past_key_values=self.past_key_values, attention_mask=attention_mask)
		
		# Update kv-cache (past_key_values)
		self.past_key_values = outputs.past_key_values
		
		return outputs.logits

# Wrapper class for tracing
class Wrapper(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model.eval()

	def forward(self, input_ids):
		return self.model(input_ids=input_ids)

# Function to handle user input dynamically via command line
def main():
	# Argument parser to allow command line input
	import argparse
	parser = argparse.ArgumentParser(description="Dynamic Core ML Model Conversion with Hugging Face Models")

	# Model input from Hugging Face
	parser.add_argument("--model", type=str, default="gpt2-large", help="Model ID from Hugging Face (e.g., gpt2, meta-llama/Meta-Llama-3-8B-Instruct)")
	
	# Compute Units options
	parser.add_argument("--compute_units", type=str, default="ALL", choices=["ALL", "CPU_AND_NE", "CPU_ONLY", "NE_ONLY"],
						help="Choose compute units for Core ML conversion (options: ALL, CPU_AND_NE, CPU_ONLY, NE_ONLY)")

	# Precision options
	parser.add_argument("--precision", type=str, default="FLOAT16", choices=["FLOAT16", "FLOAT32", "INT8"],
						help="Choose compute precision (options: FLOAT16, FLOAT32, INT8)")

	# Parse the arguments
	args = parser.parse_args()

	# Model input
	model_id = args.model

	# Select the compute units based on user input
	compute_units_mapping = {
		"ALL": ct.ComputeUnit.ALL,
		"CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
		"CPU_ONLY": ct.ComputeUnit.CPU_ONLY
	}
	compute_units = compute_units_mapping[args.compute_units]

	# Select the compute precision based on user input
	compute_precision_mapping = {
		"FLOAT16": ct.precision.FLOAT16,
		"FLOAT32": ct.precision.FLOAT32
	}
	compute_precision = compute_precision_mapping[args.precision]

	# Load the model dynamically from Hugging Face using AutoModelForCausalLM
	print(f"Loading model {model_id}...")
	torch_model = StatefulModel(model_id)

	# Example input tensors
	shape = (1, 128)
	input_ids = np.random.randint(0, 128, shape)
	input_ids = torch.tensor(input_ids, dtype=torch.int32)

	# Wrapping the model
	to_jit = Wrapper(torch_model.eval())

	# Trace the model
	with torch.no_grad():
		output_jit = to_jit(input_ids)

	# Define input and output types for Core ML conversion
	coreml_input_types = [ct.TensorType(
		name="input_ids",
		shape=ct.Shape(shape=shape),
		dtype=np.int32,
	)]
	coreml_output_types = [ct.TensorType(name="logits", dtype=np.float16)]

	# Trace the model using TorchScript
	traced_model = torch.jit.trace(to_jit, [input_ids])

	# Convert the traced model to Core ML
	print(f"Converting the model {model_id} to Core ML with precision {args.precision} and compute units {args.compute_units}...")
	fp16_stateful_mlmodel = ct.convert(
		traced_model,
		inputs=coreml_input_types,
		outputs=coreml_output_types,
		minimum_deployment_target=ct.target.iOS18,
		compute_precision=compute_precision,
		compute_units=compute_units
	)

	# Save the converted model
	mlpackage_name = f"{model_id.replace('/', '-')}-{args.precision}-CoreML.mlpackage"
	fp16_stateful_mlmodel.save(mlpackage_name)
	print(f"Model saved as {mlpackage_name}")

	# Metadata
	architecture = "AutoModelForCausalLM"  # More dynamic architecture
	user_defined_metadata = {
		"co.huggingface.exporters.name": model_id,
		"co.huggingface.exporters.task": "text-generation",
		"co.huggingface.exporters.architecture": architecture,
		"co.huggingface.exporters.framework": "pytorch",
		"co.huggingface.exporters.precision": args.precision,
	}

	# Add metadata to the Core ML model
	fp16_stateful_mlmodel._spec.description.metadata.userDefined.update(user_defined_metadata)

	# Print model card
	card = f"""
	This repository contains a Core ML conversion of [{model_id}](https://hf.co/{model_id}) with the following characteristics:

	- Sequence length: {128}, fixed.
	- Precision: {args.precision}.
	- Compute Units: {args.compute_units}.

	Please, check the [original model card](https://hf.co/{model_id}) for additional details on the model.
	"""
	print(user_defined_metadata)
	print(card)


# Entry point for the script
if __name__ == "__main__":
	main()