# SPDX-License-Identifier: MIT
"""Transformer building blocks shared by the denoiser and the text encoder.

Layout convention: every sequence tensor is **batch-first** ``(batch, length, channels)``.
Upstream's MDM runs PyTorch's ``nn.Transformer*`` modules sequence-first (``batch_first=False``),
which costs a permute at every boundary; here the layout is fixed once and the permutes vanish.
Attention runs on MLX's fused ``scaled_dot_product_attention`` with boolean masks that are
``True`` where a key may be attended to (PyTorch's ``key_padding_mask`` is the opposite).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _heads(x: mx.array, num_heads: int) -> mx.array:
    batch, length, dim = x.shape
    return x.reshape(batch, length, num_heads, dim // num_heads).transpose(0, 2, 1, 3)


def _merge(x: mx.array) -> mx.array:
    batch, heads, length, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * head_dim)


def attention(q: mx.array, k: mx.array, v: mx.array, num_heads: int, mask: mx.array | None) -> mx.array:
    """Multi-head attention core on ``(B, L, D)`` projections; ``mask`` broadcasts to ``(B, H, Lq, Lk)``."""
    head_dim = q.shape[-1] // num_heads
    out = mx.fast.scaled_dot_product_attention(_heads(q, num_heads), _heads(k, num_heads), _heads(v, num_heads),
                                               scale=1.0 / math.sqrt(head_dim), mask=mask)
    return _merge(out)


class SelfAttention(nn.Module):
    """``nn.MultiheadAttention`` used as self-attention: PyTorch's packed ``in_proj`` is kept as one
    ``nn.Linear`` (one GEMM for q, k and v, as in ``_in_projection_packed``)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # Split on the contiguous last axis (strided views), not after a transpose.
        q, k, v = mx.split(self.in_proj(x), 3, axis=-1)
        return self.out_proj(attention(q, k, v, self.num_heads, mask))


class CrossAttention(nn.Module):
    """``nn.MultiheadAttention`` used as cross-attention. PyTorch splits the packed in-projection
    into ``w_q`` and ``w_kv`` for this case; the port stores that split (``q_proj``/``kv_proj``)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, memory: mx.array, mask: mx.array | None = None) -> mx.array:
        k, v = mx.split(self.kv_proj(memory), 2, axis=-1)
        return self.out_proj(attention(self.q_proj(x), k, v, self.num_heads, mask))


class TransformerDecoderLayer(nn.Module):
    """``torch.nn.TransformerDecoderLayer`` with ``norm_first=False`` (post-norm), GELU (exact erf)
    feed-forward, LayerNorm eps 1e-5; dropout is inactive at inference."""

    def __init__(self, dim: int, num_heads: int, ff_size: int, eps: float = 1e-5):
        super().__init__()
        self.self_attn = SelfAttention(dim, num_heads)
        self.multihead_attn = CrossAttention(dim, num_heads)
        self.linear1 = nn.Linear(dim, ff_size)
        self.linear2 = nn.Linear(ff_size, dim)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.norm3 = nn.LayerNorm(dim, eps=eps)

    def __call__(self, x, memory, memory_mask=None, self_mask=None):
        x = self.norm1(x + self.self_attn(x, self_mask))
        x = self.norm2(x + self.multihead_attn(x, memory, memory_mask))
        return self.norm3(x + self.linear2(nn.gelu(self.linear1(x))))


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, dim: int, num_heads: int, ff_size: int):
        super().__init__()
        self.layers = [TransformerDecoderLayer(dim, num_heads, ff_size) for _ in range(num_layers)]

    def __call__(self, x, memory, memory_mask=None, self_mask=None):
        for layer in self.layers:
            x = layer(x, memory, memory_mask, self_mask)
        return x


class SiLU(nn.Module):
    """Parameter-free placeholder that keeps list indices aligned with ``nn.Sequential`` in the checkpoint."""

    def __call__(self, x):
        return nn.silu(x)


def sincos_table(length: int, dim: int) -> mx.array:
    """Upstream ``PositionalEncoding.pe`` (``(length, dim)``), computed like PyTorch in float32.
    Only a fallback: the loader uses the table stored in the checkpoint, which is bit-identical to
    what upstream computes at start-up."""
    import numpy as np

    position = np.arange(length, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, dim, 2, dtype=np.float32) * np.float32(-np.log(10000.0) / dim))
    pe = np.zeros((length, dim), np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return mx.array(pe)
