# SPDX-License-Identifier: MIT
"""Hyper-parameters of the two released Kirin motion generators.

Both are MDM-style transformer *decoders* (``arch=trans_dec``) over the 407-dimensional animal
motion features, conditioned through cross-attention on a memory built from the diffusion
timestep, the DistilBERT text tokens and (for ``kirin_text_image``) a DINOv3 image feature.
The values below are read from each model's ``args.json`` and checked at load time, so a
checkpoint with a different configuration fails loudly instead of running wrongly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FPS = 20  # AiM3D motion frame rate (upstream generate.py)
MAX_FRAMES = 196  # upstream generate.py caps motion_length * FPS here
NUM_JOINTS = 34  # unified SMAL-style animal skeleton
FEATURE_DIM = 407  # 4 root + 33*3 ric + 33*6 rot + 34*3 vel + 4 foot contact
TEXT_DIM = 768  # DistilBERT hidden size
IMAGE_DIM = 768  # DINOv3 ViT-B/16 [CLS] feature
POS_EMBED_LEN = 5000  # upstream PositionalEncoding max_len (pos_embed_max_len)
DEFAULT_GUIDANCE = 2.5  # upstream generate.py --guidance_param

MODELS = ("kirin_text_image", "kirin_text")
ALIASES = {"text_image": "kirin_text_image", "image": "kirin_text_image", "text": "kirin_text"}


def canonical_name(name: str) -> str:
    """``kirin_text_image`` / ``kirin_text`` (aliases ``text_image``, ``image``, ``text``)."""
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Unknown Kirin model {name!r}; choose one of {', '.join(MODELS)}.")
    return key


@dataclass(frozen=True)
class ModelSpec:
    name: str
    latent_dim: int = 512
    num_layers: int = 8
    num_heads: int = 4
    ff_size: int = 1024
    diffusion_steps: int = 50
    noise_schedule: str = "cosine"
    sigma_small: bool = True
    image_condition: bool = False
    cond_mask_prob: float = 0.2

    @property
    def supports_guidance(self) -> bool:
        # upstream parser_util: guidance is forced to 1 for models trained without condition dropout
        return self.cond_mask_prob > 0


SPECS = {
    "kirin_text_image": ModelSpec("kirin_text_image", image_condition=True),
    "kirin_text": ModelSpec("kirin_text", image_condition=False),
}

# args.json values every supported checkpoint must have (anything else needs code changes).
REQUIRED_ARGS = {
    "arch": "trans_dec", "dataset": "animal", "text_encoder_type": "bert", "emb_policy": "concat",
    "emb_trans_dec": False, "unconstrained": False, "multi_target_cond": False, "context_len": 0,
    "pred_len": 0, "lambda_target_loc": 0.0,
}


def get_spec(name: str) -> ModelSpec:
    return SPECS[canonical_name(name)]


def spec_from_args(name: str, args_path: str | Path) -> ModelSpec:
    """The spec of ``name``, verified against a checkpoint's ``args.json``."""
    spec = get_spec(name)
    args = json.loads(Path(args_path).read_text())
    for key, value in REQUIRED_ARGS.items():
        if args.get(key, value) != value:
            raise ValueError(f"{args_path}: {key}={args.get(key)!r} is not supported (expected {value!r}).")
    checks = {
        "latent_dim": spec.latent_dim, "layers": spec.num_layers, "diffusion_steps": spec.diffusion_steps,
        "noise_schedule": spec.noise_schedule, "sigma_small": spec.sigma_small,
        "image_condition": spec.image_condition, "cond_mask_prob": spec.cond_mask_prob,
    }
    for key, value in checks.items():
        if args.get(key) != value:
            raise ValueError(f"{args_path}: {key}={args.get(key)!r} does not match the {spec.name} "
                             f"configuration ({value!r}).")
    return spec
