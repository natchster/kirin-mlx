"""Weight-free unit tests: every block against an independent NumPy derivation."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from kirin_mlx import layers
from kirin_mlx.config import FEATURE_DIM, SPECS
from kirin_mlx.diffusion import cosine_betas, make_schedule
from kirin_mlx.distilbert import TransformerBlock
from kirin_mlx.mdm import MDM
from kirin_mlx.motion import recover_from_ric

RNG = np.random.default_rng(0)


def _np(a):
    return np.array(a, dtype=np.float64)


def np_layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def np_gelu(x):
    from math import erf

    return x * 0.5 * (1 + np.vectorize(erf)(x / math.sqrt(2)))


def np_linear(linear, x):
    return x @ _np(linear.weight).T + _np(linear.bias)


def np_attention(q, k, v, heads, keep=None):
    b, lq, d = q.shape
    lk = k.shape[1]
    hd = d // heads
    q = q.reshape(b, lq, heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(b, lk, heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(b, lk, heads, hd).transpose(0, 2, 1, 3)
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
    if keep is not None:
        scores = np.where(keep[:, None, None, :], scores, -np.inf)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    weights = scores / scores.sum(-1, keepdims=True)
    return (weights @ v).transpose(0, 2, 1, 3).reshape(b, lq, d)


def np_decoder_layer(layer, x, memory, keep):
    d = x.shape[-1]
    qkv = np_linear(layer.self_attn.in_proj, x)
    sa = np_attention(qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:], layer.self_attn.num_heads)
    x = np_layer_norm(x + np_linear(layer.self_attn.out_proj, sa), _np(layer.norm1.weight), _np(layer.norm1.bias), 1e-5)
    q = np_linear(layer.multihead_attn.q_proj, x)
    kv = np_linear(layer.multihead_attn.kv_proj, memory)
    ca = np_attention(q, kv[..., :d], kv[..., d:], layer.multihead_attn.num_heads, keep)
    x = np_layer_norm(x + np_linear(layer.multihead_attn.out_proj, ca), _np(layer.norm2.weight), _np(layer.norm2.bias),
                      1e-5)
    ff = np_linear(layer.linear2, np_gelu(np_linear(layer.linear1, x)))
    return np_layer_norm(x + ff, _np(layer.norm3.weight), _np(layer.norm3.bias), 1e-5)


def _randomize(module):
    """Non-trivial LayerNorm parameters, so a swapped weight/bias would be caught."""
    from mlx.utils import tree_flatten, tree_unflatten

    params = {k: mx.array(RNG.standard_normal(v.shape).astype(np.float32) * 0.2 + (1.0 if k.endswith("norm1.weight")
                                                                                       else 0.0))
              for k, v in tree_flatten(module.parameters())}
    module.update(tree_unflatten(list(params.items())))
    return module


def test_decoder_layer_matches_numpy():
    layer = _randomize(layers.TransformerDecoderLayer(64, 4, 96))
    x = RNG.standard_normal((2, 7, 64)).astype(np.float32)
    memory = RNG.standard_normal((2, 5, 64)).astype(np.float32)
    keep = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], bool)
    got = layer(mx.array(x), mx.array(memory), mx.array(keep)[:, None, None, :])
    np.testing.assert_allclose(np.array(got), np_decoder_layer(layer, x, memory, keep), rtol=2e-4, atol=2e-4)


def test_masked_memory_rows_are_ignored():
    """A padded memory token, whatever its value, must not change the output."""
    layer = _randomize(layers.TransformerDecoderLayer(32, 4, 64))
    x = mx.array(RNG.standard_normal((1, 4, 32)).astype(np.float32))
    memory = RNG.standard_normal((1, 3, 32)).astype(np.float32)
    keep = mx.array([[True, True, False]])[:, None, None, :]
    other = memory.copy()
    other[0, 2] = 1e3
    np.testing.assert_allclose(np.array(layer(x, mx.array(memory), keep)), np.array(layer(x, mx.array(other), keep)),
                               atol=1e-5)


def test_distilbert_block_matches_numpy():
    block = _randomize(TransformerBlock(48, 4, 80))
    x = RNG.standard_normal((2, 6, 48)).astype(np.float32)
    keep = np.array([[1, 1, 1, 1, 0, 0], [1] * 6], bool)
    got = block(mx.array(x), mx.array(keep)[:, None, None, :])
    a = block.attention
    sa = np_attention(np_linear(a.q_lin, x), np_linear(a.k_lin, x), np_linear(a.v_lin, x), a.num_heads, keep)
    h = np_layer_norm(np_linear(a.out_lin, sa) + x, _np(block.sa_layer_norm.weight), _np(block.sa_layer_norm.bias),
                      1e-12)
    ff = np_linear(block.ffn.lin2, np_gelu(np_linear(block.ffn.lin1, h)))
    ref = np_layer_norm(ff + h, _np(block.output_layer_norm.weight), _np(block.output_layer_norm.bias), 1e-12)
    np.testing.assert_allclose(np.array(got), ref, rtol=2e-4, atol=2e-4)


def test_sincos_table():
    table = np.array(layers.sincos_table(50, 16))
    pos = np.arange(50)[:, None]
    div = np.exp(np.arange(0, 16, 2) * -(math.log(10000.0) / 16))
    np.testing.assert_allclose(table[:, 0::2], np.sin(pos * div), atol=2e-6)
    np.testing.assert_allclose(table[:, 1::2], np.cos(pos * div), atol=2e-6)


def test_cosine_schedule_and_posterior():
    betas = cosine_betas(50)
    abar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    assert betas[0] == pytest.approx(1 - abar(1 / 50) / abar(0), rel=1e-12)
    s = make_schedule(50)
    assert s.num_steps == 50 and s.sigma[0] == 0 and (s.sigma[1:] > 0).all()
    ac = np.cumprod(1 - betas)
    ac_prev = np.append(1.0, ac[:-1])
    np.testing.assert_allclose(s.alphas_cumprod, ac, rtol=1e-12)  # SpacedDiffusion with all steps = base process
    np.testing.assert_allclose(s.coef_x0, betas * np.sqrt(ac_prev) / (1 - ac), rtol=1e-6)
    np.testing.assert_allclose(s.coef_xt, (1 - ac_prev) * np.sqrt(1 - betas) / (1 - ac), rtol=1e-6)
    var = betas * (1 - ac_prev) / (1 - ac)
    np.testing.assert_allclose(s.sigma[1:], np.sqrt(var[1:]), rtol=1e-6)
    # posterior mean is exact for x_0: mu(x_0, x_t = sqrt(abar) x_0) at t=0 returns x_0
    assert s.coef_x0[0] + s.coef_xt[0] * math.sqrt(ac[0]) == pytest.approx(1.0, rel=1e-6)


def np_recover_from_ric(data, joints):
    """A loop-by-frame NumPy transcription of upstream recover_root_rot_pos + recover_from_ric."""
    t_len = data.shape[0]
    angle, pos = 0.0, np.zeros(3)
    out = np.zeros((t_len, joints, 3))
    for t in range(t_len):
        if t > 0:
            angle += data[t - 1, 0]
            vel = np.array([data[t - 1, 1], 0.0, data[t - 1, 2]])
            # The yaw "quaternion" is (cos a, 0, sin a, 0): HumanML3D stores half-angles, so
            # qrot(qinv(q), v) rotates v by -2a about +Y (written out here).
            c, s = math.cos(2 * angle), math.sin(2 * angle)
            pos = pos + np.array([c * vel[0] - s * vel[2], 0.0, s * vel[0] + c * vel[2]])
        root = np.array([pos[0], data[t, 3], pos[2]])
        ric = data[t, 4:4 + (joints - 1) * 3].reshape(joints - 1, 3)
        c, s = math.cos(2 * angle), math.sin(2 * angle)
        rotated = np.stack([c * ric[:, 0] - s * ric[:, 2], ric[:, 1], s * ric[:, 0] + c * ric[:, 2]], -1)
        out[t, 0] = root
        out[t, 1:] = rotated + np.array([pos[0], 0.0, pos[2]])
    return out


def test_recover_from_ric_matches_numpy():
    data = RNG.standard_normal((2, 25, FEATURE_DIM)).astype(np.float32) * 0.3
    got = np.array(recover_from_ric(mx.array(data)))
    assert got.shape == (2, 25, 34, 3)
    for b in range(2):
        np.testing.assert_allclose(got[b], np_recover_from_ric(data[b].astype(np.float64), 34), atol=2e-5)


def test_unconditional_memory_follows_upstream_mask_cond():
    """mask_cond zeroes the DistilBERT features before embed_text (-> bias) and the image token after
    embed_image (-> 0)."""
    mdm = MDM(SPECS["kirin_text_image"])
    hidden = mx.array(RNG.standard_normal((2, 5, 768)).astype(np.float32))
    keep = mx.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=mx.bool_)
    image = mx.array(RNG.standard_normal((2, 768)).astype(np.float32))
    prefix, suffix, mask = mdm.memory(hidden, keep, image)
    u_prefix, u_suffix, u_mask = mdm.memory(hidden, keep, image, uncond=True)
    assert prefix.shape == (2, 1, 512) and suffix.shape == (2, 5, 512) and mask.shape == (2, 7)
    np.testing.assert_allclose(np.array(prefix[:, 0]), np.array(mdm.embed_image(image)), atol=1e-5)
    np.testing.assert_array_equal(np.array(u_prefix), 0)
    bias = np.broadcast_to(np.array(mdm.embed_text.bias), (2, 5, 512))
    np.testing.assert_allclose(np.array(u_suffix), bias, atol=1e-6)
    np.testing.assert_array_equal(np.array(mask), np.array(u_mask))  # padding mask unchanged
    np.testing.assert_array_equal(np.array(mask[:, :2]), True)  # image + timestep tokens always visible
    _, _, text_only = mdm.memory(hidden, keep, None)
    assert text_only.shape == (2, 6)


def test_text_model_rejects_image():
    with pytest.raises(ValueError, match="not image-conditioned"):
        MDM(SPECS["kirin_text"]).memory(mx.zeros((1, 2, 768)), mx.ones((1, 2), dtype=mx.bool_), mx.zeros((1, 768)))


def test_quantized_layer_keeps_shape():
    layer = layers.TransformerDecoderLayer(64, 4, 128)
    nn.quantize(layer, group_size=64, bits=8)
    out = layer(mx.zeros((1, 3, 64)), mx.zeros((1, 2, 64)))
    assert out.shape == (1, 3, 64)
