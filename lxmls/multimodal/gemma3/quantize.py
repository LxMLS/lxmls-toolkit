# Simple row-wise symmetric quantization
import torch

from lxmls.multimodal.gemma3.config import get_model_config
from lxmls.multimodal.gemma3.model import Gemma3ForMultimodalLM


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs per-row symmetric quantization on a weight tensor.

    Args:
        weight (torch.Tensor): The input float32 weight tensor (e.g., from a Linear layer).

    Returns:
        A tuple containing:
        - torch.Tensor: The quantized weight tensor (int8).
        - torch.Tensor: The per-row scaling factors (float32).
    """
    # Calculate the maximum absolute value for each row (per-output-channel)
    # Add a small epsilon to avoid division by zero for rows that are all zeros
    abs_max = torch.max(torch.abs(weight), dim=1, keepdim=True)[0] + 1e-6

    # Calculate the scaling factor for each row
    # The scaler is what you multiply the int8 value by to get the float back
    scaler = abs_max / 127.0

    # Quantize the weights
    # Divide by the scaler, round to the nearest integer, and cast to int8
    quantized_w = (weight / scaler).round().to(torch.int8)

    # The scaler tensor should be 1D
    scaler = scaler.squeeze()

    return quantized_w, scaler


# Instantiate a model with QUANT=TRUE to get the expected parameter names
config = get_model_config()
config.tokenizer = "data/vlm/gemma3/tokenizer.model"
config.quant = True
quantized_model = Gemma3ForMultimodalLM(config)
unquantized_checkpoint_path = "data/vlm/gemma3/model.ckpt"
quantized_checkpoint_path = "data/vlm/gemma3/quantized/model.ckpt"

print("Loading unquantized state dictionary...")
unquantized_state_dict = torch.load(unquantized_checkpoint_path, map_location="cpu")

# If your weights are inside a nested dictionary like {'model_state_dict': ...}
if "model_state_dict" in unquantized_state_dict:
    unquantized_state_dict = unquantized_state_dict["model_state_dict"]

quantized_state_dict = {}
print("Starting quantization process...")

for name, param in unquantized_state_dict.items():
    # Identify the weights that need to be quantized.
    # Based on your code, these are the '.weight' parameters of
    # Linear and Embedding layers.
    if name.startswith("siglip_vision_model"):
        quantized_state_dict[name] = param
        continue

    if name.endswith(".weight") and ("proj" in name or "embedder" in name or "mlp" in name):
        print(f"Quantizing layer: {name}")

        # Apply the quantization function
        quantized_w, scaler = quantize_weight(param)

        # Add the new quantized weight to the state dict
        quantized_state_dict[name] = quantized_w

        # Add the new scaler parameter to the state dict
        scaler_name = name.replace(".weight", ".weight_scaler")
        quantized_state_dict[scaler_name] = scaler

    else:
        # Copy all other parameters (biases, norms, etc.) directly
        quantized_state_dict[name] = param

print("\nQuantization complete.")

# Save the new checkpoint
final_output = {"model_state_dict": quantized_state_dict}
torch.save(final_output, quantized_checkpoint_path)
print(f"✅ Quantized model state_dict saved to: {quantized_checkpoint_path}")

# Verify quantization
print("\nVerifying by loading into a quantized model instance...")
config = get_model_config()
config.tokenizer = "data/vlm/gemma3/tokenizer.model"
config.quant = True
verification_model = Gemma3ForMultimodalLM(config)
verification_model.load_state_dict(quantized_state_dict)
print("Successfully loaded quantized state dict into model.")
