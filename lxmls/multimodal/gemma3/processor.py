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

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import sentencepiece
import torch
from PIL import Image

from lxmls.multimodal.gemma3.siglip_vision.config import DEFAULT_IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD

_BEGIN_IMAGE_TOKEN = 255999
_END_IMAGE_TOKEN = 256000

CROPPED_IMAGE_PREFIX: str = "here is the original image"
CROPPED_IMAGE_FILLER: str = "and here are some crops to help you see better"


def preprocess_images_for_siglip_vision(
    images: Sequence[Image.Image], image_size=DEFAULT_IMAGE_SIZE
) -> list[torch.Tensor]:
    """Preprocesses a list of PIL images for Siglip vision model using only PyTorch and PIL.

    Args:
        images: A sequence of PIL Image objects.
        image_size: The target size for resizing the images.

    Returns:
        A sequence of torch.Tensor objects, each of shape (C, H, W).
    """
    processed_images = []

    mean_tensor = torch.tensor(IMAGE_MEAN, dtype=torch.float32).reshape(3, 1, 1)
    std_tensor = torch.tensor(IMAGE_STD, dtype=torch.float32).reshape(3, 1, 1)

    for image in images:
        # Resize image
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

        # Convert to NumPy and ensure float32 type
        image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]

        # Convert to PyTorch tensor and rearrange channels
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [H, W, C] → [C, H, W]

        # Normalize
        image_tensor = (image_tensor - mean_tensor) / std_tensor

        # Clip the values to [-1, 1]
        image_tensor = torch.clamp(image_tensor, -1, 1)

        processed_images.append(image_tensor)

    return processed_images


@dataclass
class TokenisationOutput:
    finalised_token_ids: torch.Tensor
    image_batch: Optional[torch.Tensor]
    batch_size: int
    min_prompt_len: int
    max_prompt_len: int
    max_seq_len: int
    image_presence_mask: Optional[torch.Tensor]


def pan_and_scan(img: Image.Image, *, min_crop_size: int = 256, max_num_crops: int = 4) -> Sequence[Image.Image]:
    """Pan and scan an image for open source.

    If the image is landscape, the crops are made horizontally and if the image is
    portrait, the crops are made vertically. The longer side of the image is split
    into [2 - max_num_crops] crops.

    Args:
        img: PIL Image object.
        min_crop_size: The minimum size of each crop.
        max_num_crops: The maximum desired number of crops to be generated.

    Returns:
        List of cropped PIL Image objects and a list of crop positions.
    """
    w, h = img.size

    # Square or landscape image.
    if w >= h:
        if w / h < 1.5:
            # return [img], [(0, 0, h, w)]
            return [img]

        # Select ideal number of crops close to the image aspect ratio and such that
        # crop_size > min_crop_size.
        num_crops_w = int(np.floor(w / h + 0.5))  # Half round up rounding.
        num_crops_w = min(int(np.floor(w / min_crop_size)), num_crops_w)

        # Make sure the number of crops is in range [2, max_num_crops].
        num_crops_w = max(2, num_crops_w)
        num_crops_w = min(max_num_crops, num_crops_w)
        num_crops_h = 1

    # Portrait image.
    else:
        if h / w < 1.5:
            # return [img], [(0, 0, h, w)]
            return [img]

        num_crops_h = int(np.floor(h / w + 0.5))
        num_crops_h = min(int(np.floor(h / min_crop_size)), num_crops_h)
        num_crops_h = max(2, num_crops_h)
        num_crops_h = min(max_num_crops, num_crops_h)
        num_crops_w = 1

    crop_size_w = int(np.ceil(w / num_crops_w))
    crop_size_h = int(np.ceil(h / num_crops_h))

    # Don't apply pan and scan if crop size is too small.
    if min(crop_size_w, crop_size_h) < min_crop_size:
        # return [img], [(0, 0, h, w)]
        return [img]

    crop_positions_w = [crop_size_w * i for i in range(num_crops_w)]
    crop_positions_h = [crop_size_h * i for i in range(num_crops_h)]

    # Generate crops.
    crops = []
    crop_positions = []
    for pos_h in crop_positions_h:
        for pos_w in crop_positions_w:
            crops.append(img.crop((pos_w, pos_h, pos_w + crop_size_w, pos_h + crop_size_h)))
            crop_positions.append((pos_h, pos_w, pos_h + crop_size_h, pos_w + crop_size_w))

    # return crops, crop_positions
    return crops


def input_preprocessor(
    raw_user_prompt: Sequence[Union[Image.Image, str]],
) -> Sequence[Union[torch.Tensor, str]]:
    """Preprocessor for Gemma3 input

    Args:
      raw_user_prompt: A list of images or strings, as provided by the user.

    Returns:
      A list of preprocessed images or strings.
    """
    preprocessed_input: list[Union[torch.Tensor, str]] = []
    for element in raw_user_prompt:
        if isinstance(element, Image.Image):
            cropped_images = pan_and_scan(element)
            preprocessed_images_cropped = preprocess_images_for_siglip_vision(cropped_images)
            preprocessed_images_uncropped = preprocess_images_for_siglip_vision([element])
            if len(preprocessed_images_cropped) == 1:
                preprocessed_input.append(preprocessed_images_uncropped[0])
            elif len(preprocessed_images_cropped) > 1:
                preprocessed_input.append(CROPPED_IMAGE_PREFIX)
                preprocessed_input.append(preprocessed_images_uncropped[0])
                preprocessed_input.append(CROPPED_IMAGE_FILLER)
                preprocessed_input.extend(preprocessed_images_cropped)
            else:
                raise ValueError("No images found in the input.")
        else:
            preprocessed_input.append(element)

    return preprocessed_input


def batch_input_preprocessor(raw_input: Sequence[Sequence[Union[Image.Image, str]]]):
    """Preprocessor for Gemma3 batch input"""
    preprocessed_input: list[Sequence[Union[torch.Tensor, str]]] = []
    for element in raw_input:
        preprocessed_input.append(input_preprocessor(element))
    return preprocessed_input


class Tokenizer:
    def __init__(self, model_path: Optional[str]):
        assert os.path.isfile(model_path), model_path

        self.sp_model = sentencepiece.SentencePieceProcessor()
        self.sp_model.Load(model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.GetPieceSize()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.boi_id: int = _BEGIN_IMAGE_TOKEN
        self.eoi_id: int = _END_IMAGE_TOKEN
        self.image_token_placeholder_id: int = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        assert isinstance(s, str)
        t = self.sp_model.EncodeAsIds(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, tokens: List[int]) -> str:
        return self.sp_model.DecodeIds(tokens)
