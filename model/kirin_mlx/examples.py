# SPDX-License-Identifier: MIT
"""Example inputs shared by the CLI, the benchmarks, the tests and the sandbox oracle.

``upstream_noise`` is the bridge to the PyTorch reference: the oracle replaces ``torch.randn`` /
``torch.randn_like`` with draws from a NumPy generator, in the order and layout upstream's
sampler requests them (``x_T``, then one ``ε`` per step, each ``[B, 407, 1, T]``). The same seed
here rebuilds that exact sequence in the MLX layout (``[B, T, 407]``), so both sides sample from
identical noise and their outputs can be compared number by number.
"""

from __future__ import annotations

import numpy as np

from .config import FEATURE_DIM, IMAGE_DIM

PROMPTS = (
    "a horse rears up on its hind legs",
    "a dog runs forward and then jumps",
    "the cat walks slowly, turns around and sits down.",
    "an elephant walks forward",
)


class NoiseSource:
    """Sequential standard-normal draws (float32) from ``np.random.default_rng(seed)``."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, shape) -> np.ndarray:
        shape = tuple(int(s) for s in shape)
        self.shapes.append(shape)
        return self.rng.standard_normal(shape, dtype=np.float32)


def upstream_noise(seed: int, batch: int, frames: int, steps: int = 50) -> np.ndarray:
    """``[steps + 1, B, T, 407]``: the noise the oracle feeds upstream's sampler, MLX layout."""
    source = NoiseSource(seed)
    draws = [source((batch, FEATURE_DIM, 1, frames)) for _ in range(steps + 1)]
    return np.stack([d[:, :, 0, :].transpose(0, 2, 1) for d in draws])


FEATURE_STD = 0.55  # element std of real AiM3D DINOv3 [CLS] features (mean ≈ 0, norm ≈ 15)


def image_feature(seed: int = 0, batch: int = 1) -> np.ndarray:
    """A synthetic stand-in for a DINOv3 [CLS] feature ``[batch, 768]`` (the real encoder is gated):
    Gaussian, at the scale of real features. ``sandbox/scripts/fetch_aim_samples.py`` fetches real ones."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((batch, IMAGE_DIM), dtype=np.float32) * np.float32(FEATURE_STD)).astype(np.float32)
