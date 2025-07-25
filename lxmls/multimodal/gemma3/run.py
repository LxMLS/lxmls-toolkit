# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import torch
from PIL import Image

from lxmls.multimodal.gemma3 import config
from lxmls.multimodal.gemma3.config import GemmaConfig
from lxmls.multimodal.gemma3.model import Gemma3ForMultimodalLM
from lxmls.multimodal.gemma3.processor import TokenisationOutput, Tokenizer, batch_input_preprocessor
from lxmls.multimodal.gemma3.utils import set_default_tensor_type


def tokenize_raw_input(
    tokenizer: Tokenizer,
    raw_input: Sequence[Sequence[Union[str, Image.Image]]],
    config: GemmaConfig,
    output_len: int,
    device: Any,
) -> TokenisationOutput:
    """
    Converts a preprocessed batch of interleaved text and image inputs into
    token IDs and an image batch suitable for gemma3 model

    Args:
        preprocessed_batch: List of lists containing strings and torch.Tensor images.
        image_token_id: Token ID to represent image placeholders.
        max_image_tokens: Number of tokens reserved for each image.
        image_size: Expected size of images (C, H, W).

    Returns:
        user_input_token_ids: Batch of token IDs with shape (B, L), where L is the max sequence length.
        image_batch: Batch of images with shape (B, N, C, H, W), where N is the max number of images.
    """
    vis_cfg = config.vision_config
    assert vis_cfg is not None

    preprocessed_batch = batch_input_preprocessor(raw_input)

    # Initialize lists to store token IDs and image tensors
    all_token_ids, all_images, prompt_lengths = [], [], []
    min_prompt_len, max_prompt_len, max_num_images = float("inf"), 0, 0

    # Iterate over each user prompt in the batch
    for prompt in preprocessed_batch:
        token_ids, images = [], []
        token_ids.append(tokenizer.bos_id)
        # Process each element in the prompt
        for element in prompt:
            if isinstance(element, str):
                # Tokenize text and add to token_ids
                tokens = tokenizer.encode(element, bos=False, eos=False)
                token_ids.extend(tokens)
            elif isinstance(element, torch.Tensor):
                # Prepend (dual endline + [BEGIN OF IMAGE])
                token_ids.extend(tokenizer.encode("\n\n", bos=False, eos=False))
                token_ids.append(tokenizer.boi_id)
                # Add image placeholder tokens
                token_ids.extend([tokenizer.image_token_placeholder_id] * vis_cfg.encoding_sequence_length)
                # Append ([END OF IMAGE] + dual endline)
                token_ids.append(tokenizer.eoi_id)
                token_ids.extend(tokenizer.encode("\n\n", bos=False, eos=False))
                # Store the image tensor
                images.append(element)
            else:
                raise ValueError("Unsupported type in prompt. Expected str or torch.Tensor.")

        # Update values
        curr_prompt_len = len(token_ids)
        prompt_lengths.append(curr_prompt_len)
        max_prompt_len = max(max_prompt_len, curr_prompt_len)
        min_prompt_len = min(min_prompt_len, curr_prompt_len)
        max_num_images = max(max_num_images, len(images))

        all_token_ids.append(token_ids)
        all_images.append(images)

    max_seq_len = max_prompt_len + output_len

    # Pad token IDs to the maximum sequence length
    finalised_token_ids = []
    for token_ids in all_token_ids:
        pad_length = max_seq_len - len(token_ids)
        padded_token_ids = token_ids + [tokenizer.pad_id] * pad_length
        finalised_token_ids.append(padded_token_ids)

    # Pad images to the maximum number of images in the batch
    image_batch = []
    image_presence_mask = []
    for images in all_images:
        # Check if all images within the current sublist have the same shape
        if images:  # Check if the sublist is not empty
            first_shape = images[0].shape
            for img in images:
                assert img.shape == first_shape, "Images within a sublist must have the same shape."
        pad_length = max_num_images - len(images)
        padded_images = images.copy()  # create a copy so the original data is not altered.
        presence_mask = [True] * len(images)

        if pad_length > 0:
            # Create a list of zero tensors for padding
            padding = [
                torch.zeros((vis_cfg.input_channels, vis_cfg.image_size, vis_cfg.image_size), device=device)
                for _ in range(pad_length)
            ]
            padded_images.extend(padding)
            presence_mask.extend([False] * pad_length)
        image_batch.append(padded_images)
        image_presence_mask.append(presence_mask)

    # Convert lists to tensors
    finalised_token_ids = torch.tensor(finalised_token_ids, dtype=torch.long, device=device)
    if max_num_images > 0:
        image_batch = torch.stack([torch.stack(images) for images in image_batch]).to(device=device, dtype=config.get_dtype())
        image_presence_mask = torch.tensor(image_presence_mask, dtype=torch.bool, device=device)
    else:
        image_batch = None
        image_presence_mask = None

    # Prepare the output
    output = TokenisationOutput(
        finalised_token_ids=finalised_token_ids,
        image_batch=image_batch,
        batch_size=len(preprocessed_batch),
        min_prompt_len=min_prompt_len,  #  type: ignore Intially set to float('inf'), but then we take a min
        max_prompt_len=max_prompt_len,
        max_seq_len=max_seq_len,
        image_presence_mask=image_presence_mask,
    )
    return output


def generate(
    model: Gemma3ForMultimodalLM,
    prompts: Sequence[Sequence[Union[str, Image.Image]]],
    device: Any,
    output_len: int = 100,
    temperature: Union[float, None] = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
) -> Sequence[str]:
    processing_result = tokenize_raw_input(model.tokenizer, prompts, model.config, output_len, device)
    batch_size = processing_result.batch_size
    finalised_token_ids = processing_result.finalised_token_ids
    image_batch = processing_result.image_batch
    min_prompt_len = processing_result.min_prompt_len
    _max_prompt_len = processing_result.max_prompt_len
    total_seq_len = processing_result.max_seq_len
    image_presence_mask = processing_result.image_presence_mask

    # Create attention mask.
    assert model.dtype is not None
    min_dtype = torch.finfo(model.dtype).min
    n_inf = torch.tensor(min_dtype, dtype=model.dtype, device=device)
    assert model.config.sliding_window_size is not None
    boolean_mask, local_boolean_mask = model.create_attention_mask(finalised_token_ids, total_seq_len)
    mask_tensor = torch.where(boolean_mask, 0, n_inf).contiguous()
    local_mask_tensor = torch.where(local_boolean_mask, 0, n_inf).contiguous()

    kv_caches = []
    for _ in range(model.config.num_hidden_layers):
        size = (batch_size, total_seq_len, model.config.num_key_value_heads, model.config.head_dim)
        dtype = model.config.get_dtype()
        k_cache = torch.zeros(size=size, dtype=dtype, device=device)
        v_cache = torch.zeros(size=size, dtype=dtype, device=device)
        kv_caches.append((k_cache, v_cache))

    input_token_ids_tensor = torch.full(
        (batch_size, min_prompt_len), model.tokenizer.pad_id, dtype=torch.int64, device=device
    )
    token_ids_tensor = finalised_token_ids.to(device)
    for i in range(batch_size):
        p = finalised_token_ids[i]
        input_token_ids_tensor[i, :min_prompt_len] = p[:min_prompt_len]

    input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64, device=device)
    prompt_mask_tensor = token_ids_tensor != model.tokenizer.pad_id
    curr_mask_tensor = mask_tensor.index_select(2, torch.atleast_1d(input_positions_tensor))
    curr_local_mask_tensor = local_mask_tensor.index_select(2, torch.atleast_1d(input_positions_tensor))
    # The first iteration produces a sequence of hidden states of shape: [B, min_prompt_len, hidden_size]
    # To predict the next token, we need to look at the hidden state corresponding to the last token: [min_prompt_len - 1]
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size).to(device)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

    # Prefill up to min_prompt_len tokens, then treat other prefill as
    # decode and ignore output.
    for i in range(total_seq_len - min_prompt_len):
        next_token_ids, _ = model(
            input_token_ids=input_token_ids_tensor,
            image_patches=image_batch,  # NOTE: Always `None` after the first iteration
            image_presence_mask=image_presence_mask,
            input_positions=input_positions_tensor,
            kv_caches=kv_caches,
            mask=curr_mask_tensor,
            output_positions=output_positions_tensor,
            temperatures=temperatures_tensor,
            top_ps=top_ps_tensor,
            top_ks=top_ks_tensor,
            local_mask=curr_local_mask_tensor,
        )
        # Check if the current generation step corresponds to a position that was part of the original prompt
        curr_prompt_mask = prompt_mask_tensor.index_select(1, torch.atleast_1d(output_index)).squeeze(dim=1)
        # Retrieve the token from the original prompt at the current position
        curr_token_ids = token_ids_tensor.index_select(1, torch.atleast_1d(output_index)).squeeze(dim=1)
        # Decide which token to use
        # if `curr_prompt_mask` is True (i.e. we are still processing the prompt) use prompt token curr_token_ids
        # else use the token generated by the model: next_token_ids
        output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(dim=1)
        # Update the 'final' copy of tokens
        token_ids_tensor.index_copy_(dim=1, index=output_index, source=output_token_ids)

        # Token just generated (or prefilled) is not part of the input
        input_token_ids_tensor = output_token_ids
        # The input position for the next iteration is set to the current output_index
        input_positions_tensor = output_index.unsqueeze(dim=-1)
        # The attention masks are updated to reflect the new seq_len for the next iteration
        curr_mask_tensor = mask_tensor.index_select(2, torch.atleast_1d(input_positions_tensor))
        curr_local_mask_tensor = (
            local_mask_tensor.index_select(2, torch.atleast_1d(input_positions_tensor)) if local_mask_tensor is not None else None
        )
        # After the first iteration, where we pass in min_prompt_len tokens, we now generate tokens one by one
        # The input to the model now becomes a single token
        # The model processes this token and produces a hidden state sequence of shape: [B, 1, hidden_size]
        # So, now to get the logits for the next token, we need to use the first (and only) hidden state available
        output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
        # duh
        output_index = output_index + 1

        # The image information is already integrated into the text embeddings in the first iteration
        image_batch = None
        image_presence_mask = None

    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
        output = tokens
        if model.tokenizer.eos_id in output:
            eos_index = output.index(model.tokenizer.eos_id)
            output = output[:eos_index]
        results.append(model.tokenizer.decode(output))

    return results


def main(args):
    # Construct the model config.
    model_config = config.get_model_config()
    model_config.dtype = args.dtype or "float32"
    model_config.quant = args.quant
    image_paths = {
        "cow_in_beach": Path(args.image_dir) / "cow_in_beach.jpg",
        "lilly": Path(args.image_dir) / "lilly.jpg",
        "sunflower": Path(args.image_dir) / "sunflower.jpg",
        "golden_test_image": Path(args.image_dir) / "test_image.jpg",
    }
    model_config.tokenizer = str(Path(args.model_dir) / "tokenizer.model")

    image = {}
    for key in image_paths:
        try:
            image[key] = Image.open(image_paths[key])  # Open local file
            # image[key].show()
        except IOError as e:
            print(f"Error loading image: {e}")
            exit()

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the model and load the weights.
    device = torch.device(args.device)
    with set_default_tensor_type(model_config.get_dtype()):
        model = Gemma3ForMultimodalLM(model_config)
        model.load_state_dict(torch.load(Path(args.model_dir) / "model.ckpt")["model_state_dict"])
        model = model.to(device).eval()
    print("Model loading done")

    # Generate text only.
    result = generate(
        model,
        [
            ["<start_of_turn>user The capital of Italy is?<end_of_turn>\n<start_of_turn>model"],
            ["<start_of_turn>user What is your purpose?<end_of_turn>\n<start_of_turn>model"],
        ],
        device,
        output_len=args.output_len,
    )

    # Print the results.
    print("======================================")
    print(f"Text only RESULT: {result}")
    print("======================================")

    # Generate golden Gemax test image.
    result = generate(
        model,
        [
            [
                "<start_of_turn>user\n",
                image["golden_test_image"],
                "Caption this image. <end_of_turn>\n<start_of_turn>model",
            ]
        ],
        device,
        output_len=args.output_len,
    )

    # Print the result.
    print("======================================")
    print(f"Golden test image RESULT: {result}")
    print("======================================")

    # Generate text and image.
    result = generate(
        model,
        [
            [
                "<start_of_turn>user\n",
                image["cow_in_beach"],
                "The name of the animal in the image is <end_of_turn>\n<start_of_turn>model",
            ]
        ],
        device,
        output_len=args.output_len,
    )

    # Print the result.
    print("======================================")
    print(f"Single image RESULT: {result}")
    print("======================================")

    # Generate interleave text and multiple images.
    result = generate(
        model,
        [
            [
                "<start_of_turn>user\nThis image",
                image["lilly"],
                "and this image",
                image["sunflower"],
                "are similar because? Give me the main reason.",
                "<end_of_turn>\n<start_of_turn>model",
            ]
        ],
        device,
        output_len=args.output_len,
    )

    # Print the result.
    print("======================================")
    print(f"Interleave images RESULT: {result}")
    print("======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to directory containing the model.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument(
        "--variant",
        type=str,
        default="4b",
        choices=["4b", "12b", "27b_v3"],
        help="Model variant.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="Device to run the model on.",
    )
    parser.add_argument("--output_len", type=int, default=24, help="Length of the output sequence.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--quant", action="store_true", help="Whether to use quantization.")
    parser.add_argument("--dtype", type=str, default=None)
    args = parser.parse_args()

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    main(args)
