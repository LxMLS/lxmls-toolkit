import logging
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)7s - %(message)s")
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)


class SiglipVisionConfig:
    def __init__(self):
        # https://huggingface.co/google/gemma-3-4b-it/blob/main/config.json#L28-L37
        self.hidden_size = 1152
        self.intermediate_size = 4304
        self.num_hidden_layers = 27
        self.num_attention_heads = 16
        self.num_channels = 3
        self.image_size = 896
        self.patch_size = 14
        self.attention_dropout = 0.0
        self.layer_norm_eps = 1e-6
        self.vision_use_head = False


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution images.
        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_height, new_width), mode="bicubic", align_corners=False
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        logger.debug(f"SiglipVisionEmbeddings.forward START - {pixel_values.shape}")
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        logger.debug(f"SiglipVisionEmbeddings.forward END - {embeddings.shape}")
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=False)

        attn_out = torch.matmul(attn_weights, value)
        attn_out = attn_out.transpose(1, 2).contiguous()
        return attn_out

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logger.debug(f"SiglipAttention.forward START - {x.shape}")
        batch_size, seq_length, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = self._attn(q, k, v, attention_mask, scaling=self.scale, dropout=0.0)
        attn_out = attn_out.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_out = self.out_proj(attn_out)

        logger.debug(f"SiglipAttention.forward END - {attn_out.shape}")
        return attn_out


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = partial(F.gelu, approximate="tanh")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"SiglipMLP.forward START - {x.shape}")
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        logger.debug(f"SiglipMLP.forward END - {x.shape}")
        return x


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logger.debug(f"SiglipEncoderLayer.forward START - {x.shape}")
        residual = x

        x = self.layer_norm1(x)
        x = self.self_attn(x, attention_mask=attention_mask)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x

        logger.debug(f"SiglipEncoderLayer.forward END - {x.shape}")
        return x


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logger.debug(f"SiglipEncoder.forward START - {inputs_embeds.shape}")
        x = inputs_embeds
        for encoder_layer in self.layers:
            x = encoder_layer(x, attention_mask)
        logger.debug(f"SiglipEncoder.forward END - {x.shape}")
        return x


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, x):
        logger.debug(f"SiglipMultiheadAttentionPoolingHead.forward START - {x.shape}")
        batch_size = x.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        x = self.attention(probe, x, x)[0]

        residual = x
        x = self.layernorm(x)
        x = residual + self.mlp(x)

        out = x[:, 0]
        logger.debug(f"SiglipMultiheadAttentionPoolingHead.forward END - {out.shape}")
        return out


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(self, pixel_values, interpolate_pos_encoding: Optional[bool] = False) -> torch.Tensor:
        logger.debug(f"SiglipVisionTransformer.forward START - {pixel_values.shape}")
        x = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        x = self.encoder(inputs_embeds=x)
        x = self.post_layernorm(x)
        if self.use_head:
            x = self.head(x)

        logger.debug(f"SiglipVisionTransformer.forward END - {x.shape}")
        return x


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False, **kwargs) -> torch.Tensor:
        logger.debug(f"SiglipVisionModel.forward START - {pixel_values.shape}")
        out = self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        logger.debug(f"SiglipVisionModel.forward END - {out.shape}")
        return out

    def load_weights(self, shards_dir: Path | str):
        logger.debug(f"Loading weights from {shards_dir}")
        shards_dir = Path(shards_dir)
        state_dict = {}
        for shard_file in sorted(list(shards_dir.glob("*.safetensors"))):
            shard = load_file(shard_file, device="cpu")
            state_dict.update(shard)
        vm_state_dict = {k[len("vision_tower.") :]: v for k, v in state_dict.items() if k.startswith("language_model.")}
        self.load_state_dict(vm_state_dict, strict=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from io import BytesIO

    import requests
    from PIL import Image
    from transformers import Gemma3ImageProcessor

    parser = ArgumentParser()
    parser.add_argument("--shards", type=str)
    parser.add_argument("--len", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    config = SiglipVisionConfig()
    model = SiglipVisionModel(config)
    model.load_weights(args.shards)

    logger.debug("Loading Gemma3ImageProcessor")
    processor = Gemma3ImageProcessor.from_pretrained("google/gemma-3-4b-it")

    logger.debug("Processing image")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = Image.open(BytesIO(requests.get(url, timeout=120).content))
    inputs = processor(images=[image], return_tensors="pt")

    logger.debug("Entering Model forward")
    model(**inputs)
