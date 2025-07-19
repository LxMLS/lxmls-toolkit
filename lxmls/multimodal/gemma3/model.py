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

import gc
import json
import os
from typing import List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from lxmls.multimodal.gemma3 import siglip_vision
from lxmls.multimodal.gemma3.config import AttentionType, GemmaConfig
from lxmls.multimodal.gemma3.processor import Tokenizer


class Sampler(nn.Module):
    def __init__(self, vocab_size: int, config: GemmaConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)
        return next_token_ids, logits


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: int = 1) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = freqs / rope_scaling_factor
    t = torch.arange(end, device=freqs.device)  #  type: ignore
    freqs = torch.outer(t, freqs).float()  #  type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.int8), requires_grad=False)
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8), requires_grad=False
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)), requires_grad=False)
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, quant: bool):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, attn_type: AttentionType):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, quant=config.quant
        )
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, quant=config.quant)
        self.query_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.key_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask

        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping

        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)
        return output


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, attn_type: AttentionType):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttention(config=config, attn_type=self.attn_type)
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_pre_ffw_norm else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffw_norm else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaTextModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)] if config.attn_types is not None else AttentionType.GLOBAL
            )
            self.layers.append(Gemma2DecoderLayer(config, attn_type))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[AttentionType, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3ForMultimodalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.dtype = config.get_dtype()
        self.config = config
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size
        self.tokenizer = Tokenizer(config.tokenizer)
        self.text_token_embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaTextModel(config)
        self.sampler = Sampler(vocab_size, config)

        assert config.vision_config is not None, "Vision configuration must be provided for multimodal Gemma 3"
        self.siglip_vision_model = siglip_vision.SiglipVisionModel(config.vision_config)

        self.mm_soft_embedding_norm = RMSNorm(config.vision_config.embedding_dim, eps=config.rms_norm_eps)
        self.mm_input_projection = Linear(config.vision_config.embedding_dim, config.hidden_size, config.quant)

        rope_lengths = config.rope_wave_length
        assert rope_lengths is not None, "rope_wave_length must be provided for multimodal Gemma3"
        defaults = {AttentionType.LOCAL_SLIDING: 10_000, AttentionType.GLOBAL: 10_000}
        self._register_freqs_cis(
            "local_freqs_cis",
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(AttentionType.LOCAL_SLIDING, defaults[AttentionType.LOCAL_SLIDING]),
        )
        self._register_freqs_cis(
            "global_freqs_cis",
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(AttentionType.GLOBAL, defaults[AttentionType.GLOBAL]),
            rope_scaling_factor=config.rope_scaling_factor,
        )

    def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1
    ):
        self.register_buffer(
            name,
            precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta, rope_scaling_factor=rope_scaling_factor),
        )

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,  # [B, L]
        image_patches: torch.Tensor,  # [B, N, C, H, W] [3, 896, 896]
        image_presence_mask: torch.Tensor,  # [B, N]
        input_positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis: dict[AttentionType, torch.Tensor] = {}
        freqs_cis[AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, input_positions)
        freqs_cis[AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, input_positions)
        hidden_states = self.text_token_embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        if image_patches is not None and self.config.vision_config is not None:
            # the input has images
            B, N, C, H, W = image_patches.shape
            # Flatten and Pass to SiglipVisionModel, and apply SiglipVisionModel Exit
            flattened_input = image_patches.reshape(B * N, C, H, W)  # [B*N, C, H, W]
            image_embeddings = self.siglip_vision_model(flattened_input)  # [B*N, U, D]
            # Multi-modal projector
            image_embeddings = self.mm_soft_embedding_norm(image_embeddings)  # [B*N, U, D]
            image_embeddings = self.mm_input_projection(image_embeddings)  # [B*N, U, model_dim]
            hidden_states = self.populate_image_embeddings(
                hidden_states.clone(),
                image_embeddings.clone(),
                input_token_ids.clone(),
                image_presence_mask.clone(),
            )

        kv_write_indices = input_positions

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
        embedder_weight = self.text_token_embedder.weight
        if self.config.quant:
            embedder_weight = embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)

        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    def populate_image_embeddings(
        self,
        hidden_states: torch.Tensor,  # [B, L, model_dim]
        image_embeddings: torch.Tensor,  # [B*N, U, model_dim]
        input_token_ids: torch.Tensor,  # [B, L]
        image_presence_mask: torch.Tensor,  # [B, N]
    ):
        batch_size, seq_len, model_dim = hidden_states.shape
        # Step 1 of 2: Fetch valid image embeddings
        # flatten indices of valid image embeddings
        valid_image_embeddings_indices = torch.nonzero(image_presence_mask.flatten(), as_tuple=False).squeeze()
        # [num_valid_images, model_dim]
        valid_image_embeddings = image_embeddings.index_select(0, valid_image_embeddings_indices)

        # Step 2 of 2: Replace image embeddings at right places.
        image_placeholder_mask = input_token_ids == self.tokenizer.image_token_placeholder_id
        image_placeholder_indices = image_placeholder_mask.flatten().nonzero(as_tuple=False).squeeze()

        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        hidden_states[image_placeholder_indices] = valid_image_embeddings.reshape(-1, self.config.hidden_size)
        return hidden_states.reshape(batch_size, seq_len, model_dim).contiguous()

    def create_attention_mask(self, input_ids: torch.Tensor, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_ids.shape[0]
        causal_mask = torch.tril(
            torch.ones((batch_size, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device)
        )
        image_token_mask = input_ids == self.tokenizer.image_token_placeholder_id
        # Pad the mask to the left with 0. This is to make sure the boundary
        # detection works correctly. Boundary (starting index of image patch) is
        # detected when the value changes from 0 to 1.
        padded_mask = nn.functional.pad(image_token_mask, (1, 0), value=0)
        # Find the boundary (starting index) of the image tokens patch.
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        # Number the boundary.
        # boundary:
        # [[False, False,  True, False, False],
        #  [False,  True, False,  True, False]]
        # numbered_boundary:
        # [[0, 0, 1, 1, 1],
        #  [0, 1, 1, 2, 2]]
        numbered_boundary = torch.cumsum(boundary, dim=-1)

        # image_token_mask:
        # [[False, False,  True,  True, False],
        #  [True,  True, False,  True, True]]
        # numbered_boundary:
        # [[0, 0, 1, 1, 1],
        #  [1, 1, 1, 2, 2]]
        # q_block_indices:
        # [[0, 0, 1, 1, 0],
        #  [1, 1, 0, 2, 2]]
        q_block_indices = image_token_mask * numbered_boundary
        kv_block_indices = q_block_indices
        # Test the equality of vertical and horizontal numbered patches
        # to create the bidirectional mask.
        bidirectional_mask = torch.logical_and(
            kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
            q_block_indices.unsqueeze(-1) > 0,
        )
        attention_mask = torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
        # The upper triangular matrix's diagonal is shifted by sliding window size
        # before doing logical 'and' with attention mask. This is to make sure the
        # local attention is within the sliding window.
        local_mask = torch.logical_and(
            attention_mask,
            torch.triu(
                torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device),
                diagonal=-(self.config.sliding_window_size - 1),
            ),
        )
        return attention_mask, local_mask

    def load_weights(self, model_path: str):
        if os.path.isfile(model_path):
            self.load_state_dict(torch.load(model_path, mmap=True, weights_only=True)["model_state_dict"], strict=False)
        else:
            index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()
