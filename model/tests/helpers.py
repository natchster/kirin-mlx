"""Shared helpers for the MLX test-suite (not named conftest: sandbox/tests has one too).

Checkpoints: ``checkpoints/`` (``kirin-mlx download``), or ``$KIRIN_CHECKPOINTS``. Golden fixtures
from the PyTorch reference: ``sandbox/golden`` (``sandbox/setup.sh``), or ``$KIRIN_GOLDEN``. Tests
needing either skip without them, or fail with ``KIRIN_REQUIRE_FIXTURES=1`` (see conftest.py).
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = Path(os.environ.get("KIRIN_CHECKPOINTS", ROOT / "checkpoints"))
GOLDEN = Path(os.environ.get("KIRIN_GOLDEN", ROOT / "sandbox" / "golden"))
MODELS = ("kirin_text_image", "kirin_text")
CASES = {
    "kirin_text_image": ("text_image", "image_only", "no_image", "max_frames", "per_sample"),
    "kirin_text": ("text", "mixed", "unguided"),
}
ALL_CASES = [(m, c) for m in MODELS for c in CASES[m]]


def checkpoint_dir(name: str) -> Path:
    return CHECKPOINTS / "kirin" / name


def have_checkpoints(name: str) -> bool:
    return (checkpoint_dir(name) / "model.pt").is_file() and (
        CHECKPOINTS / "distilbert-base-uncased" / "model.safetensors").is_file()


def fresh_model(name: str, dtype: str = "float32"):
    """A new, uncached model, for tests that change it; skips without the checkpoints."""
    import kirin_mlx as kirin

    if not have_checkpoints(name):
        pytest.skip(f"{name} checkpoints missing (run `kirin-mlx download`)")
    return kirin.load(name, checkpoint_dir(name), dtype=dtype)


@cache
def load_model(name: str, dtype: str = "float32"):
    return fresh_model(name, dtype)


def load_golden(name: str, case: str) -> dict[str, np.ndarray]:
    path = GOLDEN / name / f"{case}.npz"
    if not path.is_file():
        pytest.skip(f"golden fixture {path} missing (made by the sandbox: sandbox/setup.sh)")
    return dict(np.load(path))


def golden_request(golden: dict) -> dict:
    """The ``generate`` arguments of a golden case (plus its noise, rebuilt from the seed)."""
    from kirin_mlx.examples import upstream_noise

    texts = [str(t) for t in golden["in.texts"]]
    frames = int(golden["in.num_frames"])
    return {"text": texts, "image_feature": golden.get("in.image_feature"), "num_frames": frames,
            "guidance": float(golden["in.guidance"]),
            "noise": upstream_noise(int(golden["in.seed"]), len(texts), frames)}


def rel_err(got, ref) -> float:
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12))
