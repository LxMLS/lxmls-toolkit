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

import torch
import torch.nn.functional as F
from torch import nn

from lxmls.multimodal.gemma3.siglip_vision.config import SiglipVisionModelConfig


class AveragePool2D(nn.Module):
    """Applies 4x4 average pooling and reshaping."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        width = int(seq_len**0.5)
        if width * width != seq_len:
            raise ValueError(f"Sequence length {seq_len} is not a perfect square. Cannot reshape to a square image.")

        # [B, 64^2, 1152] -> [B, 1152, 64^2] -> [B, 1152, 64, 64]
        x = x.transpose(1, 2).reshape(batch_size, channels, width, width)
        # [B, 1152, 64, 64] -> [B, 1152, 16, 16]
        x = F.avg_pool2d(x, kernel_size=4, stride=4)
        # [B, 1152, 64, 64] -> [B, 1152, 256] -> [B, 256, 1152]
        x = x.flatten(2).transpose(1, 2)
        return x


class SiglipAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Key, Query, Value projections
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=True)

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=True)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Project inputs to key, query, value
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for multi-head attention
        k = k.transpose(1, 2)  # [B, H, S, head_dim]
        q = q.transpose(1, 2)  # [B, H, S, head_dim]
        v = v.transpose(1, 2)  # [B, H, S, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back to [B, S, H, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Apply output projection
        output = self.o_proj(attn_output)

        return output


class SiglipMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def gelu_tanh(self, x):
        return (0.5 * x) * (
            1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu_tanh(x)
        x = self.fc2(x)
        return x


class SiglipEncoderBlock(nn.Module):
    def __init__(self, config: SiglipVisionModelConfig):
        super().__init__()
        self.self_attn = SiglipAttention(config.embedding_dim, config.num_attention_heads, config.head_dim)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config.embedding_dim, config.intermediate_size)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)

    def forward(self, x):
        # Pre-LN
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = x + residual  # Residual connection *after* LayerNorm

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual  # Residual connection *after* LayerNorm
        return x


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionModelConfig):
        super().__init__()

        self.patch_embedding = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.embedding_dim,
            kernel_size=config.conv2d_patch_size,
            stride=config.conv2d_patch_size,
            padding=0,
            bias=config.embedding_use_bias,
        )
        self.num_patches = (config.image_size // config.conv2d_patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, config.embedding_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

        self.encoder_blocks = nn.ModuleList(SiglipEncoderBlock(config=config) for _ in range(config.num_hidden_layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim, config.layer_norm_eps)
        self.avg_pool = AveragePool2D(config)
        self.config = config

    @torch.inference_mode
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, config.input_channels, config.image_size, config.image_size]

        # Embed the image according to SiplipVisionEmbeddings
        x = self.patch_embedding(pixel_values)
        # [B, C, H W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)

        position_ids = self.position_ids.to(pixel_values.device)
        x = x + self.position_embedding(position_ids)

        for block in self.encoder_blocks:
            x = block(x)  # [B, H*W, E], where E is the embedding_dim (1152)
        x = self.final_norm(x)

        # siglip exit
        return self.avg_pool(x)
