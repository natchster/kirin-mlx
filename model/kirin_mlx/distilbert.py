# SPDX-License-Identifier: MIT
# MLX port of DistilBERT (Sanh et al., 2019) as implemented by Hugging Face transformers
# (models/distilbert/modeling_distilbert.py, Apache-2.0), forward pass only.
"""The frozen text encoder Kirin conditions on: ``distilbert-base-uncased``'s last hidden state.

Upstream (``model/BERT/BERT_encoder.py``) runs ``AutoModel(...)(**tokenizer(texts, padding=True))``
and passes ``last_hidden_state`` plus the attention mask to the denoiser. This is the same
6-layer post-norm encoder: learned position embeddings, LayerNorm eps 1e-12, 12 heads, exact-GELU
feed-forward of width 3072. Padded keys are masked out of attention; padded query rows are still
computed (they are masked again in the denoiser's cross-attention, as upstream).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .layers import attention


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, max_positions: int, dim: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_positions, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=1e-12)

    def __call__(self, input_ids: mx.array) -> mx.array:
        positions = mx.arange(input_ids.shape[1])
        return self.LayerNorm(self.word_embeddings(input_ids) + self.position_embeddings(positions))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        return self.out_lin(attention(self.q_lin(x), self.k_lin(x), self.v_lin(x), self.num_heads, mask))


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x):
        return self.lin2(nn.gelu(self.lin1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = MultiHeadSelfAttention(dim, num_heads)
        self.sa_layer_norm = nn.LayerNorm(dim, eps=1e-12)
        self.ffn = FFN(dim, hidden_dim)
        self.output_layer_norm = nn.LayerNorm(dim, eps=1e-12)

    def __call__(self, x, mask):
        x = self.sa_layer_norm(self.attention(x, mask) + x)
        return self.output_layer_norm(self.ffn(x) + x)


class Transformer(nn.Module):
    def __init__(self, num_layers: int, dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.layer = [TransformerBlock(dim, num_heads, hidden_dim) for _ in range(num_layers)]

    def __call__(self, x, mask):
        for block in self.layer:
            x = block(x, mask)
        return x


class DistilBert(nn.Module):
    """``DistilBertModel``: ``(input_ids [B, L] int, attention_mask [B, L] bool) -> [B, L, 768]``."""

    def __init__(self, vocab_size=30522, max_positions=512, dim=768, num_layers=6, num_heads=12, hidden_dim=3072):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, max_positions, dim)
        self.transformer = Transformer(num_layers, dim, num_heads, hidden_dim)

    @classmethod
    def from_config(cls, path: str | Path) -> DistilBert:
        config = json.loads(Path(path).read_text())
        if config.get("model_type") != "distilbert" or config.get("activation") != "gelu" or config.get(
                "sinusoidal_pos_embds"):
            raise ValueError(f"{path} is not a supported DistilBERT configuration")
        return cls(config["vocab_size"], config["max_position_embeddings"], config["dim"], config["n_layers"],
                   config["n_heads"], config["hidden_dim"])

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        mask = attention_mask[:, None, None, :]  # keys a query may attend to
        return self.transformer(self.embeddings(input_ids), mask)
