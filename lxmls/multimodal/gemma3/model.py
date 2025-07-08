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
"""Inference-only Gemma model implementation."""

import gc
import json
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

import lxmls.multimodal.gemma3.config as gemma_config
from lxmls.multimodal.gemma3 import preprocessor, tokenizer
from lxmls.multimodal.gemma3.siglip_vision import siglip_vision_model


class Sampler(nn.Module):
    def __init__(self, vocab_size: int, config: gemma_config.GemmaConfig):
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
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
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
    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
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
        local_mask: torch.Tensor = None,
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
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
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


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.attn_type = gemma_config.AttentionType.GLOBAL
        self.self_attn = GemmaAttention(config=config, attn_type=self.attn_type)
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, quant=config.quant
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
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


class GemmaModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaDecoderLayer(config))
            elif config.architecture in (
                gemma_config.Architecture.GEMMA_2,
                gemma_config.Architecture.GEMMA_3,
            ):
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma2DecoderLayer(config, attn_type))
            else:
                raise ValueError(f"Unknown architecture: {config.architecture}")
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
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


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModel(config)
        self.sampler = Sampler(vocab_size, config)

        # Pre-compute rotary embedding table.
        if config.architecture == gemma_config.Architecture.GEMMA_3:
            if config.rope_wave_length is None:
                raise ValueError("rope_wave_length must be provided for Gemma3.")

            rope_lengths = config.rope_wave_length
            defaults = {
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 10_000,
            }

            for attn_type, name in [
                (gemma_config.AttentionType.LOCAL_SLIDING, "local_freqs_cis"),
                (gemma_config.AttentionType.GLOBAL, "global_freqs_cis"),
            ]:
                theta = rope_lengths.get(attn_type, defaults[attn_type])
                self._register_freqs_cis(name, head_dim, max_seq_len, theta=theta)

        else:
            self._register_freqs_cis("freqs_cis", head_dim, max_seq_len)

    def _register_freqs_cis(self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000):
        self.register_buffer(name, precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta))

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = {}

        if self.config.architecture == gemma_config.Architecture.GEMMA_3:
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, input_positions)
            freqs_cis[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, input_positions)
        else:
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = self.freqs_cis.index_select(0, input_positions)
            freqs_cis[gemma_config.AttentionType.GLOBAL] = self.freqs_cis.index_select(0, input_positions)

        kv_write_indices = input_positions

        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(-1)
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Gemma model."""
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads, self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len), self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), self.tokenizer.pad_id, dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, : len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        local_mask_tensor = (
            mask_tensor
            + torch.tril(
                torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
                diagonal=-self.config.sliding_window_size,
            )
            if self.config.sliding_window_size
            else None
        )
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = (
            local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
        )
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                local_mask=curr_local_mask_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            curr_local_mask_tensor = (
                local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
            )
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
            output_index = output_index + 1

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]) : len(prompt_tokens[i]) + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        # If a string was provided as input, return a string as output.
        return results[0] if is_str_prompt else results

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


class Gemma3ForMultimodalLM(nn.Module):
    """Gemma3 model for multimodal causal LM."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.dtype = config.get_dtype()
        assert config.architecture == gemma_config.Architecture.GEMMA_3
        self.config = config
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.text_token_embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModel(config)
        self.sampler = Sampler(vocab_size, config)

        if config.vision_config is None:
            raise ValueError("vision_config must be provided for Gemma3.")
        self.siglip_vision_model = siglip_vision_model.SiglipVisionModel(config.vision_config)
        # transformer/embedder/mm_soft_embedding_norm
        self.mm_soft_embedding_norm = RMSNorm(config.vision_config.embedding_dim, eps=config.rms_norm_eps)
        # transformer/embedder/mm_input_projection
        self.mm_input_projection = Linear(config.vision_config.embedding_dim, config.hidden_size, config.quant)

        if config.rope_wave_length is None:
            raise ValueError("rope_wave_length must be provided for Gemma3.")
        rope_lengths = config.rope_wave_length
        defaults = {gemma_config.AttentionType.LOCAL_SLIDING: 10_000, gemma_config.AttentionType.GLOBAL: 10_000}
        self._register_freqs_cis(
            "local_freqs_cis",
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(
                gemma_config.AttentionType.LOCAL_SLIDING, defaults[gemma_config.AttentionType.LOCAL_SLIDING]
            ),
        )
        self._register_freqs_cis(
            "global_freqs_cis",
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(gemma_config.AttentionType.GLOBAL, defaults[gemma_config.AttentionType.GLOBAL]),
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
        input_token_ids: torch.Tensor,  # B x L
        image_patches: torch.Tensor,  # B x N x C x H x W (3x896x896)
        image_presence_mask: torch.Tensor,  # B x N
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
        freqs_cis = {}
        freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, input_positions)
        freqs_cis[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, input_positions)
        hidden_states = self.text_token_embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        if image_patches is not None and self.config.vision_config is not None:
            # the input has images
            B, N, C, H, W = image_patches.shape
            # Flatten and Pass to SiglipVisionModel, and apply SiglipVisionModel Exit
            flattened_input = image_patches.reshape(B * N, C, H, W)  # (B*N)xCxHxW
            image_embeddings = self.siglip_vision_model(flattened_input)  # (B*N)xUxD
            image_embeddings = self.mm_soft_embedding_norm(image_embeddings)  # (B*N) x U x D
            image_embeddings = self.mm_input_projection(image_embeddings)  # (B*N) x U x model_dim
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
        hidden_states: torch.Tensor,  # B x L x model_dim
        image_embeddings: torch.Tensor,  # (B*N) x U x model_dim
        input_token_ids: torch.Tensor,  # B x L
        image_presence_mask: torch.Tensor,  # B x N
    ):
        batch_size, seq_len, model_dim = hidden_states.shape
        # Step 1 of 2: Fetch valid image embeddings
        # flatten indices of valid image embeddings
        valid_image_embeddings_indices = torch.nonzero(image_presence_mask.flatten(), as_tuple=False).squeeze()
        # num_valid_images x model_dim
        valid_image_embeddings = image_embeddings.index_select(0, valid_image_embeddings_indices)

        # Step 2 of 2: Replace image embeddings at right places.
        image_placeholder_mask = input_token_ids == self.tokenizer.image_token_placeholder_id
        image_placeholder_indices = image_placeholder_mask.flatten().nonzero(as_tuple=False).squeeze()

        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        hidden_states[image_placeholder_indices] = valid_image_embeddings.reshape(-1, self.config.hidden_size)
        return hidden_states.reshape(batch_size, seq_len, model_dim).contiguous()

    def create_attention_mask(self, input_ids: torch.Tensor, sequence_length: int):
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

    def generate(
        self,
        prompts: Sequence[Sequence[Union[str, Image.Image]]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Sequence[str]:
        """Generates responses for given prompts using Gemma model."""
        # Inference only.
        processing_result = preprocessor.tokenize_raw_input(self.tokenizer, prompts, self.config, output_len, device)
        batch_size = processing_result["batch_size"]
        user_input_token_ids = processing_result["user_input_token_ids"]
        image_batch = processing_result["image_batch"]
        min_prompt_len = processing_result["min_prompt_len"]
        _max_prompt_len = processing_result["max_prompt_len"]
        total_seq_len = processing_result["max_seq_len"]
        image_presence_mask = processing_result["image_presence_mask"]

        # Create attention mask.
        min_dtype = torch.finfo(self.dtype).min
        if self.config.sliding_window_size is None:
            raise ValueError("gemma 3 model requires sliding_window size")
        boolean_mask, local_boolean_mask = self.create_attention_mask(user_input_token_ids, total_seq_len)
        mask_tensor = torch.where(
            boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)
        ).contiguous()
        local_mask_tensor = torch.where(
            local_boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)
        ).contiguous()

        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, total_seq_len, self.config.num_key_value_heads, self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        input_token_ids_tensor = torch.full(
            (batch_size, min_prompt_len), self.tokenizer.pad_id, dtype=torch.int64, device=device
        )
        token_ids_tensor = user_input_token_ids.to(device)
        for i in range(batch_size):
            p = user_input_token_ids[i]
            input_token_ids_tensor[i, :min_prompt_len] = p[:min_prompt_len]

        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64, device=device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(total_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                image_patches=image_batch,
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
            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            curr_local_mask_tensor = (
                local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
            )
            output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
            output_index = output_index + 1
            image_batch = None
            image_presence_mask = None

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            output = tokens
            if self.tokenizer.eos_id in output:
                eos_index = output.index(self.tokenizer.eos_id)
                output = output[:eos_index]
            results.append(self.tokenizer.decode(output))

        return results

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
