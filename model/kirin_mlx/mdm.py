# SPDX-License-Identifier: MIT
# MLX port of Kirin's denoiser (briannlongzhao/kirin, motion_generation/model/mdm.py), which is
# MDM (Tevet et al., "Human Motion Diffusion Model", MIT) with DiP-style BERT conditioning and an
# image token. Original: Copyright (c) 2022 Guy Tevet; Copyright (c) 2026 Brian Nlong Zhao, MIT.
"""The motion denoiser: ``x_t [B, T, 407]`` + timestep + condition memory -> ``x_0`` prediction.

Released configuration (``arch=trans_dec``, ``emb_policy=concat``, ``text_encoder_type=bert``)::

    x_t ──poseEmbedding──► + sinusoidal PE ──► 8 × post-norm TransformerDecoderLayer ──poseFinal──► x̂_0
                                                     ▲ cross-attention
    memory = [ image token? | timestep token | text tokens (DistilBERT → embed_text) ]

Classifier-free guidance needs an *unconditional* memory too. Upstream's ``mask_cond`` zeroes the
DistilBERT features **before** ``embed_text`` (so every text token becomes ``embed_text.bias``)
and zeroes the image token **after** ``embed_image``; the timestep token and the padding mask are
unchanged. :meth:`MDM.memory` reproduces both, so a guided step can run the conditional and the
unconditional halves as one batch of ``2B``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import FEATURE_DIM, IMAGE_DIM, POS_EMBED_LEN, TEXT_DIM, ModelSpec
from .layers import SiLU, TransformerDecoder, sincos_table


class TimestepEmbedder(nn.Module):
    """``pe[t] -> Linear -> SiLU -> Linear`` (``time_embed`` is an ``nn.Sequential`` upstream)."""

    def __init__(self, dim: int):
        super().__init__()
        self.time_embed = [nn.Linear(dim, dim), SiLU(), nn.Linear(dim, dim)]

    def __call__(self, pe_rows: mx.array) -> mx.array:
        x = pe_rows
        for layer in self.time_embed:
            x = layer(x)
        return x


class InputProcess(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.poseEmbedding = nn.Linear(FEATURE_DIM, dim)

    def __call__(self, x):
        return self.poseEmbedding(x)


class OutputProcess(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.poseFinal = nn.Linear(dim, FEATURE_DIM)

    def __call__(self, x):
        return self.poseFinal(x)


class MDM(nn.Module):
    """Parameter names follow the PyTorch checkpoint (see :mod:`kirin_mlx.weights`)."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        dim = spec.latent_dim
        self.input_process = InputProcess(dim)
        self.seqTransDecoder = TransformerDecoder(spec.num_layers, dim, spec.num_heads, spec.ff_size)
        self.embed_timestep = TimestepEmbedder(dim)
        self.embed_text = nn.Linear(TEXT_DIM, dim)
        if spec.image_condition:
            self.embed_image = nn.Linear(IMAGE_DIM, dim)
        self.output_process = OutputProcess(dim)
        # Fixed sinusoidal table (not a parameter: the leading underscore keeps it out of
        # `parameters()`). The loader replaces it with the table stored in the checkpoint.
        self._pe = sincos_table(POS_EMBED_LEN, dim)
        self._dtype = mx.float32  # compute dtype of the activations (set by Kirin.set_compute_dtype)

    def set_position_table(self, table: mx.array) -> None:
        if table.ndim == 3:  # PyTorch buffer shape (max_len, 1, dim)
            table = table[:, 0, :]
        if table.shape != self._pe.shape:
            raise ValueError(f"position table has shape {table.shape}, expected {self._pe.shape}")
        self._pe = table.astype(mx.float32)

    @property
    def compute_dtype(self) -> mx.Dtype:
        return self._dtype

    @compute_dtype.setter
    def compute_dtype(self, dtype: mx.Dtype) -> None:
        self._dtype = dtype

    # -- conditioning -------------------------------------------------------------------------
    def timestep_embedding(self, timesteps: mx.array) -> mx.array:
        """``[S] int -> [S, D]``: the timestep token for each of ``S`` timesteps."""
        return self.embed_timestep(self._pe[timesteps].astype(self._dtype))

    def memory(self, text_hidden: mx.array, text_mask: mx.array, image_feature: mx.array | None,
               uncond: bool = False) -> tuple[mx.array, mx.array]:
        """The condition memory **without** its timestep token, and its key mask.

        ``text_hidden [B, L, 768]`` / ``text_mask [B, L]`` (True on real tokens) from DistilBERT;
        ``image_feature [B, 768]`` or ``None`` (text-only mode, upstream ``--no_image``).
        Returns ``(prefix [B, P, D], suffix [B, L, D], mask [B, P + 1 + L])`` where the timestep
        token goes between ``prefix`` (the image token, ``P`` = 0 or 1) and ``suffix`` (text)."""
        dtype = self._dtype
        if uncond:  # upstream mask_cond(enc_text, force_mask=True): zeros before embed_text
            text = mx.broadcast_to(self.embed_text(mx.zeros((1, 1, TEXT_DIM), dtype)),
                                   (*text_hidden.shape[:2], self.spec.latent_dim))
        else:
            text = self.embed_text(text_hidden.astype(dtype))
        batch = text_hidden.shape[0]
        if image_feature is not None:
            if not self.spec.image_condition:
                raise ValueError(f"{self.spec.name} is not image-conditioned; use kirin_text_image for image input.")
            image = self.embed_image(image_feature.astype(dtype))[:, None, :]
            if uncond:  # mask_cond after embed_image: the token becomes zeros
                image = mx.zeros_like(image)
            prefix = image
        else:
            prefix = mx.zeros((batch, 0, self.spec.latent_dim), text.dtype)
        keep = mx.ones((batch, prefix.shape[1] + 1), dtype=mx.bool_)  # image + timestep tokens
        return prefix, text, mx.concatenate([keep, text_mask.astype(mx.bool_)], axis=1)

    # -- denoiser -----------------------------------------------------------------------------
    def __call__(self, x: mx.array, time_token: mx.array, prefix: mx.array, suffix: mx.array,
                 memory_mask: mx.array) -> mx.array:
        """``x [B, T, 407]`` (any float dtype), ``time_token [B, D]`` -> ``x̂_0 [B, T, 407]``."""
        dtype = self._dtype
        memory = mx.concatenate([prefix, time_token.astype(dtype)[:, None, :], suffix], axis=1)
        h = self.input_process(x.astype(dtype))
        h = h + self._pe[: x.shape[1]].astype(dtype)
        h = self.seqTransDecoder(h, memory, memory_mask[:, None, None, :])
        return self.output_process(h)
