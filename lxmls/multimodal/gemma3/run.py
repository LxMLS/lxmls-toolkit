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
import contextlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lxmls.multimodal.gemma3 import config
from lxmls.multimodal.gemma3 import model as gemma3_model


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def main(args):
    # Data Dir
    data_dir = Path(args.ckpt).parent

    # Construct the model config.
    model_config = config.get_model_config()
    model_config.dtype = "float32"
    model_config.quant = args.quant
    image_paths = {
        "cow_in_beach": data_dir / "images/cow_in_beach.jpg",
        "lilly": data_dir / "images/lilly.jpg",
        "sunflower": data_dir / "images/sunflower.JPG",
        "golden_test_image": data_dir / "images/test_image.jpg",
    }
    model_config.tokenizer = str(data_dir / "tokenizer.model")

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
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma3_model.Gemma3ForMultimodalLM(model_config)
        model.load_state_dict(torch.load(args.ckpt)["model_state_dict"])
        # model.load_weights(args.ckpt)
        model = model.to(device).eval()
    print("Model loading done")

    # Generate text only.
    result = model.generate(
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
    result = model.generate(
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
    result = model.generate(
        [
            [
                "<start_of_turn>user\n",
                image["cow_in_beach"],
                ("The name of the animal in the image is <end_of_turn>\n<start_of_turn>model"),
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
    result = model.generate(
        [
            [
                "<start_of_turn>user\nThis image",
                image["lilly"],
                "and this image",
                image["sunflower"],
                "are similar because? <end_of_turn>\n<start_of_turn>model",
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
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
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
    args = parser.parse_args()
    main(args)
