# SPDX-License-Identifier: MIT
"""Load the released checkpoints into the MLX modules, strictly.

Sources:

* ``<model>/model.pt``: the denoiser's PyTorch state dict, read with :mod:`kirin_mlx.torch_pickle`
  (NumPy only, no code execution);
* ``distilbert-base-uncased/model.safetensors``: the text encoder (``mx.load``).

PyTorch -> MLX name mapping is mechanical:

* ``nn.MultiheadAttention`` packs q/k/v in raw ``in_proj_weight``/``in_proj_bias`` parameters.
  The self-attention keeps them packed as an ``nn.Linear`` (``self_attn.in_proj``); the
  cross-attention stores the split PyTorch itself makes for that case (``q_proj``/``kv_proj``);
* the two copies of the fixed sinusoidal table (``*.sequence_pos_encoder.pe``) become the model's
  position table instead of parameters (upstream deletes them and recomputes the same values);
* DistilBERT's ``distilbert.`` prefix is dropped, and its unused masked-LM head (``vocab_*``) too.

Every load fails on a missing, unexpected or misshaped tensor.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from .torch_pickle import load_state_dict

FORMAT_VERSION = "1"
PE_KEYS = ("sequence_pos_encoder.pe", "embed_timestep.sequence_pos_encoder.pe")


def mdm_state(model_pt: str | Path, latent_dim: int) -> tuple[dict[str, mx.array], mx.array]:
    """``(MLX-named denoiser parameters, position table)`` from a Kirin ``model.pt``."""
    torch_state = load_state_dict(model_pt)
    tables = [torch_state.pop(k) for k in PE_KEYS if k in torch_state]
    if not tables:
        raise KeyError(f"{model_pt}: no sequence_pos_encoder.pe table")
    if any(not np.array_equal(t, tables[0]) for t in tables):
        raise ValueError(f"{model_pt}: the two position tables differ")
    state: dict[str, mx.array] = {}
    for name, value in torch_state.items():
        if ".multihead_attn.in_proj_" in name:
            base, kind = name.rsplit(".in_proj_", 1)  # kind: weight | bias
            state[f"{base}.q_proj.{kind}"] = mx.array(value[:latent_dim])
            state[f"{base}.kv_proj.{kind}"] = mx.array(value[latent_dim:])
        elif ".self_attn.in_proj_" in name:
            base, kind = name.rsplit(".in_proj_", 1)
            state[f"{base}.in_proj.{kind}"] = mx.array(value)
        else:
            state[name] = mx.array(value)
    return state, mx.array(tables[0][:, 0, :])


def distilbert_state(safetensors_path: str | Path) -> dict[str, mx.array]:
    state = {}
    for name, value in mx.load(str(safetensors_path)).items():
        if name.startswith("vocab_"):  # masked-LM head, unused by AutoModel / Kirin
            continue
        if not name.startswith("distilbert."):
            raise KeyError(f"Unexpected DistilBERT tensor {name!r}")
        state[name.removeprefix("distilbert.")] = value
    return state


def assign(module, params: dict[str, mx.array], what: str) -> None:
    """Load ``params`` into ``module`` strictly: no missing, unexpected or misshaped tensors."""
    expected = dict(tree_flatten(module.parameters()))
    unexpected = sorted(set(params) - set(expected))
    missing = sorted(set(expected) - set(params))
    if unexpected or missing:
        raise KeyError(f"{what}: {len(missing)} missing tensors (e.g. {missing[:3]}), "
                       f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
    for key, value in params.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"{what}: shape mismatch for {key}: checkpoint {value.shape}, model {expected[key].shape}")
    module.load_weights([(k, v.astype(mx.float32)) for k, v in params.items()], strict=True)


def save_converted(path: str | Path, name: str, mdm_params: dict, text_params: dict, pe: mx.array, mean, std,
                   vocab: str, dtype=mx.float32) -> Path:
    """One self-contained file: denoiser + text encoder parameters, position table, feature
    statistics, and the tokenizer vocabulary (in the metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"mdm.{k}": v.astype(dtype) for k, v in mdm_params.items()}
    arrays |= {f"text_encoder.{k}": v.astype(dtype) for k, v in text_params.items()}
    arrays["pe"] = pe.astype(mx.float32)
    arrays["stats.mean"] = mx.array(np.asarray(mean, np.float32))
    arrays["stats.std"] = mx.array(np.asarray(std, np.float32))
    mx.save_safetensors(str(path), arrays, metadata={"model": name, "format": FORMAT_VERSION, "vocab": vocab})
    return path


def read_converted(path: str | Path):
    """``(name, mdm params, text params, pe, mean, std, vocab)`` from :func:`save_converted`."""
    arrays, metadata = mx.load(str(path), return_metadata=True)
    if metadata.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a kirin-mlx v{FORMAT_VERSION} weight file.")
    mdm = {k.removeprefix("mdm."): v for k, v in arrays.items() if k.startswith("mdm.")}
    text = {k.removeprefix("text_encoder."): v for k, v in arrays.items() if k.startswith("text_encoder.")}
    return (metadata["model"], mdm, text, arrays["pe"], np.array(arrays["stats.mean"]), np.array(arrays["stats.std"]),
            metadata["vocab"])
