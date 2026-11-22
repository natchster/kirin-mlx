# SPDX-License-Identifier: MIT
# MLX port of Kirin's generation path (briannlongzhao/kirin, motion_generation/generate.py,
# utils/sampler_util.py, diffusion/gaussian_diffusion.py). Original code MIT; see NOTICE.
"""The Kirin motion generator on MLX: text and/or image in, 3D animal joint motion out.

    import kirin_mlx as kirin
    model = kirin.load("kirin_text_image", "checkpoints/kirin/kirin_text_image")
    out = model.generate("a horse rears up on its hind legs", num_samples=3, seed=0)
    out["joints"]      # [3, 120, 34, 3] world-space joint positions at 20 fps (6 s by default)

One call does what upstream ``generate.py`` does for one prompt: DistilBERT encodes the text
once, 50 guided denoising steps run from Gaussian noise, and the features are de-normalized
and integrated into joint positions. Differences from upstream are performance-only:

* the conditional and unconditional passes of classifier-free guidance run as **one batch**
  (upstream's ``ClassifierFreeSampleModel`` calls the network twice per step);
* the timestep tokens of all 50 steps are computed once per model, and each denoising step is
  one ``mx.compile``-d graph; the 50 steps are queued lazily and evaluated together;
* identical prompts in a batch are encoded once.

The random numbers come from MLX, so a seed does not reproduce a PyTorch run. To replay one,
pass upstream's noise (``noise=[x_T, ε_49, …, ε_0]``); the parity tests do exactly that.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import hub
from . import weights as W
from .config import (
    DEFAULT_GUIDANCE,
    FEATURE_DIM,
    FPS,
    IMAGE_DIM,
    MAX_FRAMES,
    POS_EMBED_LEN,
    ModelSpec,
    canonical_name,
    get_spec,
    spec_from_args,
)
from .diffusion import make_schedule
from .distilbert import DistilBert
from .mdm import MDM
from .motion import recover_from_ric
from .tokenizer import WordPieceTokenizer

DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def resolve_dtype(dtype) -> mx.Dtype:
    """``"float32"`` / ``"float16"`` / ``"bfloat16"`` (or the MLX dtype itself) -> MLX dtype."""
    if isinstance(dtype, mx.Dtype):
        if dtype in DTYPES.values():
            return dtype
    elif isinstance(dtype, str) and dtype in DTYPES:
        return DTYPES[dtype]
    raise ValueError(f"Unsupported compute dtype {dtype!r}; choose one of {', '.join(DTYPES)} "
                     "(for INT8 weights, call .quantize(8) on a loaded model).")


class Condition:
    """Everything a denoising step needs besides ``x_t`` and ``t`` (built by :meth:`Kirin.condition`).

    ``prefix``/``suffix``/``mask`` are the memory around the timestep token; with guidance they
    hold the conditional rows followed by the unconditional ones (``2B`` rows)."""

    def __init__(self, prefix, suffix, mask, batch: int, guided: bool, scale: float):
        self.prefix, self.suffix, self.mask = prefix, suffix, mask
        self.batch, self.guided, self.scale = batch, guided, scale


class Kirin(nn.Module):
    """A Kirin generator (``kirin_text_image`` or ``kirin_text``) with its text encoder."""

    def __init__(self, spec: ModelSpec, mean: np.ndarray, std: np.ndarray, tokenizer: WordPieceTokenizer,
                 text_encoder: DistilBert | None = None):
        super().__init__()
        self.spec = spec
        self.mdm = MDM(spec)
        self.text_encoder = text_encoder or DistilBert()
        self.tokenizer = tokenizer
        self._mean = mx.array(np.asarray(mean, np.float32))
        self._std = mx.array(np.asarray(std, np.float32))
        schedule = make_schedule(spec.diffusion_steps, spec.noise_schedule)
        self.schedule = schedule
        self._coef_x0 = mx.array(schedule.coef_x0)
        self._coef_xt = mx.array(schedule.coef_xt)
        self._sigma = mx.array(schedule.sigma)
        self._time_table = None
        self._compiled_step = None
        self._quant_bits: int | None = None

    # -- precision ----------------------------------------------------------------------------
    @property
    def dtype(self) -> mx.Dtype:
        return self.mdm.compute_dtype

    def set_compute_dtype(self, dtype) -> Kirin:
        """FP32 (default) or FP16/BF16 for the denoiser. The text encoder, the diffusion update,
        de-normalization and joint recovery stay FP32."""
        dtype = resolve_dtype(dtype)
        self.mdm.set_dtype(dtype)  # floating parameters only: packed quantized weights keep their dtype
        self.mdm.compute_dtype = dtype
        self._materialize()
        return self

    def set_dtype(self, dtype, predicate=None) -> Kirin:
        """Same as :meth:`set_compute_dtype` (``nn.Module.set_dtype`` would also cast the text encoder)."""
        if predicate is not None:
            raise TypeError("Kirin.set_dtype takes no predicate: the precision applies to the denoiser.")
        return self.set_compute_dtype(dtype)

    def quantize(self, bits: int = 8, group_size: int = 64) -> Kirin:
        """Experimental: affine-quantize the denoiser's transformer layers and output projection."""
        if self._quant_bits is not None:
            raise RuntimeError("The model is already quantized; load it again to re-quantize.")
        nn.quantize(self.mdm.seqTransDecoder, group_size=group_size, bits=bits)
        nn.quantize(self.mdm.output_process, group_size=group_size, bits=bits)
        self._quant_bits = bits
        self._materialize()
        return self

    @property
    def quantization(self) -> int | None:
        return self._quant_bits

    def update(self, parameters: dict, strict: bool = True) -> Kirin:
        super().update(parameters, strict=strict)
        self._materialize()
        return self

    def _materialize(self) -> None:
        """Evaluate lazily cast/quantized parameters now (otherwise the first compiled trace would
        capture the casts and redo them on every call), and drop everything derived from them."""
        mx.eval(self.parameters())
        self._time_table = None
        self._compiled_step = None

    # -- conditioning -------------------------------------------------------------------------
    def encode_text(self, texts: list[str]) -> tuple[mx.array, mx.array]:
        """DistilBERT ``last_hidden_state [B, L, 768]`` (FP32) and attention mask ``[B, L]``, padded to
        the longest text like upstream's ``tokenizer(texts, padding=True)``. Duplicates run once."""
        texts = [str(t) for t in texts]
        unique = list(dict.fromkeys(texts))  # first-seen order
        ids, mask = self.tokenizer(unique)
        ids, mask = mx.array(ids), mx.array(mask)
        hidden = self.text_encoder(ids, mask)
        if len(unique) != len(texts):
            row = {text: i for i, text in enumerate(unique)}
            index = mx.array([row[t] for t in texts], dtype=mx.int32)
            hidden, mask = hidden[index], mask[index]
        return hidden, mask

    def time_tokens(self) -> mx.array:
        """``[S, D]``: the timestep token of every diffusion step (cached until the weights change)."""
        if self._time_table is None:
            self._time_table = self.mdm.timestep_embedding(mx.arange(self.spec.diffusion_steps))
            mx.eval(self._time_table)
        return self._time_table

    def condition(self, text_hidden: mx.array, text_mask: mx.array, image_feature: mx.array | None = None,
                  guidance: float = DEFAULT_GUIDANCE) -> Condition:
        guided = guidance != 1
        if guided and not self.spec.supports_guidance:
            raise ValueError(f"{self.spec.name} was trained without condition dropout; use guidance=1.")
        prefix, suffix, mask = self.mdm.memory(text_hidden, text_mask, image_feature)
        if guided:
            u_prefix, u_suffix, _ = self.mdm.memory(text_hidden, text_mask, image_feature, uncond=True)
            prefix = mx.concatenate([prefix, u_prefix])
            suffix = mx.concatenate([suffix, u_suffix])
            mask = mx.concatenate([mask, mask])
        return Condition(prefix, suffix, mask, text_hidden.shape[0], guided, float(guidance))

    # -- one network evaluation (x̂_0) and one sampler step -------------------------------------
    def _x0(self, x, t, prefix, suffix, mask, scale):
        time = mx.broadcast_to(self.time_tokens()[t], (prefix.shape[0], self.spec.latent_dim))
        if prefix.shape[0] == x.shape[0]:  # unguided
            return self.mdm(x, time, prefix, suffix, mask).astype(mx.float32)
        out = self.mdm(mx.concatenate([x, x]), time, prefix, suffix, mask).astype(mx.float32)
        out_cond, out_uncond = mx.split(out, 2)
        return out_uncond + scale * (out_cond - out_uncond)  # upstream ClassifierFreeSampleModel

    def predict_x0(self, x: mx.array, t: int, cond: Condition) -> mx.array:
        """Guided ``x̂_0`` for ``x [B, T, 407]`` (FP32) at timestep ``t`` (eager, for tests and tools)."""
        return self._x0(x, mx.array(t, dtype=mx.int32), cond.prefix, cond.suffix, cond.mask,
                        mx.array(cond.scale, dtype=mx.float32))

    def _step(self, x, t, noise, prefix, suffix, mask, scale):
        x0 = self._x0(x, t, prefix, suffix, mask, scale)
        mean = self._coef_x0[t] * x0 + self._coef_xt[t] * x  # q_posterior_mean_variance
        return mean + self._sigma[t] * noise  # sigma[0] = 0: no noise on the last step

    def step(self, x: mx.array, t: int, noise: mx.array, cond: Condition, compiled: bool = True) -> mx.array:
        """``x_t -> x_{t-1}`` (one ``p_sample``); ``noise`` is ``ε`` (ignored at ``t = 0``)."""
        self.time_tokens()
        fn = self._step
        if compiled:
            if self._compiled_step is None:
                self._compiled_step = mx.compile(self._step)
            fn = self._compiled_step
        return fn(x, mx.array(t, dtype=mx.int32), noise, cond.prefix, cond.suffix, cond.mask,
                  mx.array(cond.scale, dtype=mx.float32))

    def sample(self, cond: Condition, num_frames: int, noise=None, key: mx.array | None = None,
               compiled: bool = True) -> mx.array:
        """``p_sample_loop``: normalized features ``[B, T, 407]`` (lazy; call ``mx.eval``)."""
        steps = self.spec.diffusion_steps
        shape = (cond.batch, num_frames, FEATURE_DIM)
        if noise is not None:
            noise = mx.array(noise, dtype=mx.float32) if not isinstance(noise, mx.array) else noise.astype(mx.float32)
            if noise.shape != (steps + 1, *shape):
                raise ValueError(f"noise has shape {noise.shape}, expected {(steps + 1, *shape)} "
                                 f"([x_T, ε_{steps - 1}, …, ε_0])")
            draws = [noise[i] for i in range(steps + 1)]
        else:
            keys = mx.random.split(key if key is not None else mx.random.key(np.random.randint(2**31)), steps + 1)
            draws = [mx.random.normal(shape, key=keys[i]) for i in range(steps + 1)]
        x = draws[0]
        for i, t in enumerate(range(steps - 1, -1, -1)):
            x = self.step(x, t, draws[i + 1], cond, compiled)
        return x

    # -- features -> joints --------------------------------------------------------------------
    def denormalize(self, features: mx.array) -> mx.array:
        return features * self._std + self._mean

    def to_joints(self, features: mx.array) -> mx.array:
        """De-normalized features ``[..., T, 407]`` -> joint positions ``[..., T, 34, 3]``."""
        return recover_from_ric(features)

    # -- the public generation call ------------------------------------------------------------
    def generate(self, text: str | list[str] | None = None, image_feature=None, num_samples: int | None = None,
                 motion_length: float = 6.0, num_frames: int | None = None, guidance: float = DEFAULT_GUIDANCE,
                 seed: int | None = None, noise=None, compiled: bool = True) -> dict:
        """Generate ``num_samples`` motions; returns NumPy ``joints [N, T, 34, 3]`` (world positions,
        20 fps), ``features [N, T, 407]`` (de-normalized), ``texts`` and ``lengths``.

        ``text``: one prompt (repeated ``num_samples`` times), one prompt per sample, or ``None``
        for image-only generation (upstream substitutes the placeholder prompt ``" "``).
        ``image_feature``: a DINOv3 ViT-B/16 [CLS] feature, ``[768]``/``[1, 768]`` (shared) or
        ``[N, 768]``; only for ``kirin_text_image``; ``None`` = text only (upstream ``--no_image``).
        ``motion_length`` seconds (``T = min(196, int(motion_length * 20))`` like upstream) or an
        explicit ``num_frames``. ``guidance`` is the classifier-free scale (1 disables it)."""
        texts, image, n = self._inputs(text, image_feature, num_samples)
        frames = int(num_frames) if num_frames is not None else min(MAX_FRAMES, int(motion_length * FPS))
        if not 1 <= frames <= POS_EMBED_LEN:
            raise ValueError(f"num_frames must be in [1, {POS_EMBED_LEN}], got {frames}")
        hidden, mask = self.encode_text(texts)
        cond = self.condition(hidden, mask, image, guidance)
        key = mx.random.key(seed) if seed is not None else None
        features = self.denormalize(self.sample(cond, frames, noise=noise, key=key, compiled=compiled))
        joints = self.to_joints(features)
        mx.eval(joints, features)
        return {"joints": np.array(joints), "features": np.array(features), "texts": texts,
                "lengths": np.full(n, frames, np.int64)}

    def _inputs(self, text, image_feature, num_samples):
        if text is None:
            text = " "
        if isinstance(text, str):
            n = 1 if num_samples is None else int(num_samples)
            texts = [text] * n
        else:
            texts = [str(t) for t in text]
            if num_samples is not None and int(num_samples) != len(texts):
                raise ValueError(f"num_samples={num_samples} but {len(texts)} prompts were given")
            n = len(texts)
        if n < 1:
            raise ValueError("num_samples must be at least 1")
        image = None
        if image_feature is not None:
            if not self.spec.image_condition:
                raise ValueError(f"{self.spec.name} is not image-conditioned; use kirin_text_image.")
            feature = np.asarray(image_feature, np.float32)
            if feature.shape == (IMAGE_DIM,):
                feature = feature[None]
            if feature.ndim != 2 or feature.shape[1] != IMAGE_DIM or feature.shape[0] not in (1, n):
                raise ValueError(f"image_feature has shape {feature.shape}; expected [768], [1, 768] or [{n}, 768]")
            image = mx.array(np.broadcast_to(feature, (n, IMAGE_DIM)))
        return texts, image, n


# -- construction ---------------------------------------------------------------------------------
def _text_encoder_dir(checkpoint_dir: Path, text_encoder_dir) -> Path:
    if text_encoder_dir is not None:
        return Path(text_encoder_dir).expanduser()
    return checkpoint_dir.parent.parent / hub.TEXT_ENCODER  # hub layout: <root>/kirin/<model>, <root>/distilbert-...


def from_checkpoints(name: str, checkpoint_dir: str | Path, text_encoder_dir: str | Path | None = None,
                     dtype="float32") -> Kirin:
    """Build from the Hugging Face folders (``checkpoints/kirin/<model>`` + ``checkpoints/distilbert-base-uncased``)."""
    name, dtype = canonical_name(name), resolve_dtype(dtype)
    root = Path(checkpoint_dir).expanduser()
    files = [root / f for f in ("args.json", "model.pt", "Mean.npy", "Std.npy")]
    encoder_root = _text_encoder_dir(root, text_encoder_dir)
    files += [encoder_root / f for f in ("config.json", "model.safetensors", "vocab.txt")]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files {missing}. Run `kirin-mlx download` (pinned revisions, "
                                "SHA-256 verified).")
    spec = spec_from_args(name, root / "args.json")
    mean = np.load(root / "Mean.npy", allow_pickle=False)
    std = np.load(root / "Std.npy", allow_pickle=False)
    model = Kirin(spec, mean, std, WordPieceTokenizer.from_file(encoder_root / "vocab.txt"),
                  DistilBert.from_config(encoder_root / "config.json"))
    params, pe = W.mdm_state(root / "model.pt", spec.latent_dim)
    W.assign(model.mdm, params, f"{name} denoiser")
    model.mdm.set_position_table(pe)
    W.assign(model.text_encoder, W.distilbert_state(encoder_root / "model.safetensors"), "DistilBERT")
    model.eval()
    return model.set_compute_dtype(dtype)


def from_converted(path: str | Path, dtype="float32") -> Kirin:
    """Build from a single file written by :func:`convert`."""
    dtype = resolve_dtype(dtype)
    name, mdm_params, text_params, pe, mean, std, vocab = W.read_converted(path)
    model = Kirin(get_spec(name), mean, std, WordPieceTokenizer(vocab))
    W.assign(model.mdm, mdm_params, f"{name} denoiser")
    model.mdm.set_position_table(pe)
    W.assign(model.text_encoder, text_params, "DistilBERT")
    model.eval()
    return model.set_compute_dtype(dtype)


def convert(name: str, checkpoint_dir: str | Path, output: str | Path, text_encoder_dir: str | Path | None = None,
            dtype="float32") -> Path:
    """Write one self-contained MLX ``.safetensors`` (denoiser, DistilBERT, statistics, vocabulary)."""
    model = from_checkpoints(name, checkpoint_dir, text_encoder_dir)
    from mlx.utils import tree_flatten

    return W.save_converted(output, model.spec.name, dict(tree_flatten(model.mdm.parameters())),
                            dict(tree_flatten(model.text_encoder.parameters())), model.mdm._pe,
                            np.array(model._mean), np.array(model._std), model.tokenizer.vocab_text,
                            resolve_dtype(dtype))


def load(name_or_path: str | Path, checkpoint_dir: str | Path | None = None, dtype="float32",
         text_encoder_dir: str | Path | None = None) -> Kirin:
    """``load("kirin_text_image", "checkpoints/kirin/kirin_text_image")`` or ``load("weights/kirin.safetensors")``."""
    if checkpoint_dir is not None:
        return from_checkpoints(str(name_or_path), checkpoint_dir, text_encoder_dir, dtype)
    path = Path(name_or_path).expanduser()
    if not path.is_file():
        try:
            name = canonical_name(str(name_or_path))
        except ValueError:
            raise FileNotFoundError(f"No converted weight file at {path} (write one with kirin_mlx.convert, "
                                    "or pass a model name and its checkpoint folder).") from None
        default = hub.checkpoint_dir(name)
        if default.is_dir():
            return from_checkpoints(name, default, text_encoder_dir, dtype)
        raise ValueError(f"load({name!r}) needs the checkpoint folder (default {default} is missing: run "
                         "`kirin-mlx download`), or a file written by convert().")
    return from_converted(path, dtype)
