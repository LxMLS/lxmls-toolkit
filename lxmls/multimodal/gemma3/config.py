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

import dataclasses
import enum
from typing import Optional, Sequence

import torch

from lxmls.multimodal.gemma3.siglip_vision.config import SiglipVisionModelConfig

# Keep a mapping from dtype strings to the supported torch dtypes.
_STR_DTYPE_TO_TORCH_DTYPE = dict(
    {
        "float16": torch.float16,
        "float": torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
)


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


@dataclasses.dataclass
class GemmaConfig:
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = "bfloat16"
    # Whether a quantized version of the model is used.
    quant: bool = False
    # The path to the model tokenizer.
    tokenizer: Optional[str] = "tokenizer/tokenizer.model"
    # The types of attention used in the layers of the model.
    attn_types: Optional[Sequence[AttentionType]] = None
    # The size of the sliding window used for local attention.
    sliding_window_size: Optional[int] = None
    # If provided, the final logits are softcapped to this value.
    final_logit_softcapping: Optional[float] = None
    # If provided, the attention logits are softcapped to this value.
    attn_logit_softcapping: Optional[float] = None
    # If provided, the query vector is normalized using the
    # inverse square root of this value instead of head_dim.
    query_pre_attn_scalar: Optional[int] = None
    # Whether to use pre mlp normalization.
    use_pre_ffw_norm: bool = False
    # Whether to use post mlp normalization.
    use_post_ffw_norm: bool = False
    # The wave length of the rotary embedding.
    rope_wave_length: dict[AttentionType, int] | None = None
    # Whether to use QK normalization in the attention blocks.
    use_qk_norm: bool = False
    # Vision model config.
    vision_config: SiglipVisionModelConfig | None = None
    # The factor by which the rope wave length is divided for global layers.
    rope_scaling_factor: int | None = None

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


def get_model_config(dtype: str = "bfloat16") -> GemmaConfig:
    return GemmaConfig(
        dtype=dtype,
        num_hidden_layers=34,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=2560,
        intermediate_size=10240,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        head_dim=256,
        attn_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        sliding_window_size=1024,
        rope_wave_length={
            AttentionType.LOCAL_SLIDING: 10_000,
            AttentionType.GLOBAL: 1_000_000,
        },
        vocab_size=262_144,
        tokenizer="tokenizer/tokenizer.model",
        use_qk_norm=True,
        vision_config=SiglipVisionModelConfig(),
        rope_scaling_factor=8,
    )
