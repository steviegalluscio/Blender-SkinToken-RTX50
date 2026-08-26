"""
Self-contained pure PyTorch implementations of diffusers modules.
Removes diffusers dependency entirely.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any, Dict

logger = logging.getLogger(__name__)

USE_PEFT_BACKEND = False


def is_torch_version(op: str, ver: str) -> bool:
    return True


def scale_lora_layers(*args, **kwargs):
    pass


def unscale_lora_layers(*args, **kwargs):
    pass


def apply_rotary_emb(x: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis
    else:
        cos = freqs_cis.cos()
        sin = freqs_cis.sin()
    
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (x_rot * sin)


class ConfigMixin:
    config_name = "config.json"
    def __init__(self):
        self._internal_dict = {}
        
    @property
    def config(self):
        return getattr(self, "_internal_dict", {})


def register_to_config(fn):
    import inspect
    def wrapper(self, *args, **kwargs):
        try:
            sig = inspect.signature(fn)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            for k, v in bound.arguments.items():
                if k != "self" and not hasattr(self, k):
                    setattr(self, k, v)
            self._internal_dict = {k: v for k, v in bound.arguments.items() if k != "self"}
        except Exception:
            for k, v in kwargs.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
        return fn(self, *args, **kwargs)
    return wrapper


class ModelMixin(nn.Module):
    def __init__(self):
        super().__init__()


def maybe_allow_in_graph(cls_or_fn):
    return cls_or_fn


def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
    return torch.randn(shape, generator=generator, device=device, dtype=dtype, layout=layout)


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate=self.approximate)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        if activation_fn == "geglu":
            act = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "gelu":
            act = GELU(dim, inner_dim, approximate="none", bias=bias)
        else:
            act = GEGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([
            act,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=bias),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return x


class LayerNorm(nn.LayerNorm):
    pass


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        w = self.weight.float() if self.weight is not None else None
        b = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(orig_type)


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
        **kwargs,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = emb.chunk(2, dim=-1)
        if scale.ndim == 2 and x.ndim == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        if self.flip_sin_to_cos:
            return torch.cat([cos_emb, sin_emb], dim=-1)
        return torch.cat([sin_emb, cos_emb], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu", out_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight: bool = True, log: bool = True, flip_sin_to_cos: bool = False, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.weight[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional[Any] = None,
        qk_norm: Optional[str] = None,
        cross_attention_norm: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim
        self.is_cross_attention = cross_attention_dim is not None

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5 if scale_qk else 1.0

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim or query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim or query_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        ])
        self.processor = processor
        self.spatial_norm = None
        self.group_norm = None
        self.residual_connection = residual_connection
        self.rescale_output_factor = rescale_output_factor
        self.only_cross_attention = only_cross_attention

        if cross_attention_norm == "layer_norm" and cross_attention_dim is not None:
            self.norm_cross = nn.LayerNorm(cross_attention_dim, eps=eps)
        else:
            self.norm_cross = None

        if qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_k = nn.RMSNorm(dim_head, eps=eps)
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

    @property
    def norm_encoder_hidden_states(self):
        return self.norm_cross

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None, out_dim=3):
        return attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.processor is not None:
            return self.processor(self, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        return hidden_states


class AttentionProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return attn(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)
