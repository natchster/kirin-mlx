# SPDX-License-Identifier: MIT
"""The upstream Kirin PyTorch generator as a reference oracle.

Builds the model exactly as upstream ``motion_generation/generate.py`` does: its own argument
parser reads ``args.json``, ``create_model_and_diffusion`` builds the MDM and the spaced
diffusion, ``load_saved_model`` loads the weights, ``ClassifierFreeSampleModel`` wraps it, and
``p_sample_loop`` samples. Two things are substituted, both outside the model code:

* **DistilBERT location.** Upstream loads ``distilbert/distilbert-base-uncased`` from the Hugging
  Face hub at run time (whatever revision is current). The oracle points ``load_bert`` at the
  pinned, SHA-256-verified copy that ``kirin_mlx.hub`` downloads. Same weights, same code.
* **Random numbers.** ``torch.randn``/``torch.randn_like`` draw from a NumPy generator during
  sampling (:class:`kirin_mlx.examples.NoiseSource`), so the MLX side can rebuild the identical
  noise from the seed (:func:`kirin_mlx.examples.upstream_noise`).

Forward hooks capture the denoiser's intermediate tensors at chosen timesteps so the MLX port can
be checked layer by layer, teacher-forced.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kirin_mlx import hub
from kirin_mlx.config import canonical_name
from kirin_mlx.examples import NoiseSource

from .paths import CHECKPOINTS, MOTION_GENERATION, UPSTREAM

UPSTREAM_PACKAGES = ("model", "utils", "diffusion", "data_loaders")
RECORD_STEPS = (49, 48, 25, 24, 1, 0)  # x_t at these t: teacher-forced steps 49->48, 25->24, 1->0
HOOK_STEP = 49  # per-layer captures at this step

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return (MOTION_GENERATION / "model" / "mdm.py").is_file()


def checkpoint_dir(name: str) -> Path:
    return hub.checkpoint_dir(name, CHECKPOINTS)


def text_encoder_dir() -> Path:
    return hub.text_encoder_dir(CHECKPOINTS)


def require_checkpoints(name: str) -> None:
    """Fail early, with the fix, when a model's checkpoint files are missing."""
    missing = [e.path for e in hub.files_for(name) if not (CHECKPOINTS / e.path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} checkpoint file(s) for {canonical_name(name)} missing under "
                                f"{CHECKPOINTS} (e.g. {missing[0]}): run `kirin-mlx download` or sandbox/setup.sh")


def upstream_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fixture_meta(generator: str, device: str, **extra) -> dict:
    """What produced a set of golden fixtures. Fixtures are only bit-reproducible with the same
    versions, and only CPU fixtures are the FP32 reference."""
    import torch
    import transformers

    meta = {"generator": generator, "device": device, "upstream_commit": upstream_commit(),
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "machine": platform.machine()}
    meta.update(extra)
    return meta


def write_meta(directory: Path, meta: dict) -> Path:
    path = Path(directory) / "meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def import_upstream() -> None:
    """Put upstream ``motion_generation/`` first on ``sys.path``. Its packages (``model``,
    ``utils``, ...) are namespace packages with generic names, so any same-named module imported
    from elsewhere is evicted first."""
    if not (MOTION_GENERATION / "model" / "mdm.py").is_file():
        raise FileNotFoundError(f"{MOTION_GENERATION} missing: run sandbox/setup.sh first")
    root = str(MOTION_GENERATION)
    if sys.path[:1] != [root]:
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    for name in list(sys.modules):
        top = name.split(".")[0]
        module = sys.modules[name]
        if top in UPSTREAM_PACKAGES:
            origin = getattr(module, "__file__", None) or next(iter(getattr(module, "__path__", []) or []), "")
            if not str(origin).startswith(root):
                del sys.modules[name]


@contextlib.contextmanager
def _argv(argv):
    saved = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = saved


def generate_args(name: str, *extra: str):
    """Upstream ``utils.parser_util.generate_args()`` for ``name`` (it reads ``args.json``)."""
    import_upstream()
    from utils.parser_util import generate_args as upstream_generate_args

    with _argv(["generate.py", "--model_path", str(checkpoint_dir(name) / "model.pt"), *extra]):
        return upstream_generate_args()


@contextlib.contextmanager
def local_text_encoder():
    """Point upstream's hard-coded ``distilbert/distilbert-base-uncased`` at the verified local copy."""
    import model.mdm as mdm_module

    original = mdm_module.load_bert
    mdm_module.load_bert = lambda _path: original(str(text_encoder_dir()))
    try:
        yield
    finally:
        mdm_module.load_bert = original


@dataclass
class Reference:
    name: str
    model: object  # MDM, or ClassifierFreeSampleModel(MDM)
    diffusion: object
    args: object
    device: str

    @property
    def mdm(self):
        return getattr(self.model, "model", self.model) if self.guided else self.model

    @property
    def guided(self) -> bool:
        return type(self.model).__name__ == "ClassifierFreeSampleModel"

    @property
    def image_condition(self) -> bool:
        return "image" in self.mdm.cond_mode


def build(name: str, device: str = "cpu", image: bool = True, guidance: float = 2.5) -> Reference:
    """The upstream model and diffusion, set up like ``generate.py``'s ``__main__``.

    ``image=False`` on ``kirin_text_image`` is upstream's ``--no_image`` (text-only mode)."""
    import torch

    name = canonical_name(name)
    require_checkpoints(name)
    extra = ["--guidance_param", str(guidance)]
    if not image and name == "kirin_text_image":
        extra.append("--no_image")
    args = generate_args(name, *extra)
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils.sampler_util import ClassifierFreeSampleModel

    with local_text_encoder(), contextlib.redirect_stdout(sys.stderr):
        model, diffusion = create_model_and_diffusion(args, None)
        # generate.py, verbatim in effect:
        args.image_condition = bool(args.image_condition or args.image_path is not None
                                    or args.image_feature_path is not None)
        if args.no_image:
            args.image_condition = False
        model.cond_mode = "text" if not args.image_condition else model.cond_mode
        load_saved_model(model, args.model_path, use_avg=args.use_ema)
        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return Reference(name, model, diffusion, args, device)


@contextlib.contextmanager
def numpy_randn(source: NoiseSource, device):
    """``torch.randn(*shape)`` / ``torch.randn_like(x)`` -> draws from ``source`` (float32)."""
    import torch

    original_randn, original_like = torch.randn, torch.randn_like

    def randn(*shape, device=None, dtype=None, **_):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        return torch.from_numpy(source(shape)).to(device=device or "cpu", dtype=dtype or torch.float32)

    def randn_like(x, **_):
        return torch.from_numpy(source(tuple(x.shape))).to(device=x.device, dtype=x.dtype)

    torch.randn, torch.randn_like = randn, randn_like
    try:
        yield
    finally:
        torch.randn, torch.randn_like = original_randn, original_like


def _np(x):
    import torch

    return x.detach().to("cpu").float().numpy() if isinstance(x, torch.Tensor) else x


@contextlib.contextmanager
def capture(ref: Reference, record: dict):
    """Record the denoiser's tensors (batch-first, ``[B, T, ...]``) at :data:`HOOK_STEP` and the
    sampler state at :data:`RECORD_STEPS`."""
    mdm = ref.mdm
    state = {"t": None, "branch": "cond"}
    handles = []

    def mdm_pre(module, args, kwargs):
        timesteps = args[1]
        y = kwargs.get("y", args[2] if len(args) > 2 else None)
        state["t"] = int(timesteps[0])
        state["branch"] = "uncond" if y.get("uncond", False) else "cond"

    def tag(key):
        return f"step{state['t']}.{state['branch']}.{key}"

    def mdm_post(module, args, kwargs, output):
        if state["t"] in RECORD_STEPS:
            record[tag("x0")] = _np(output)[:, :, 0, :].transpose(0, 2, 1)

    def at_hook_step():
        return state["t"] == HOOK_STEP

    def input_hook(module, args, output):
        if at_hook_step():
            record[tag("input_process")] = _np(output).transpose(1, 0, 2)

    def decoder_pre(module, args, kwargs):
        if at_hook_step():
            tgt = kwargs.get("tgt", args[0] if args else None)
            memory = kwargs.get("memory", args[1] if len(args) > 1 else None)
            record[tag("decoder_in")] = _np(tgt).transpose(1, 0, 2)
            record[tag("memory")] = _np(memory).transpose(1, 0, 2)
            mask = kwargs.get("memory_key_padding_mask")
            if mask is not None:
                record[tag("memory_keep")] = ~_np(mask).astype(bool)

    def layer_hook(index):
        def hook(module, args, output):
            if at_hook_step():
                record[tag(f"layer{index}")] = _np(output).transpose(1, 0, 2)
        return hook

    handles.append(mdm.register_forward_pre_hook(mdm_pre, with_kwargs=True))
    handles.append(mdm.register_forward_hook(mdm_post, with_kwargs=True))
    handles.append(mdm.input_process.register_forward_hook(input_hook))
    handles.append(mdm.seqTransDecoder.register_forward_pre_hook(decoder_pre, with_kwargs=True))
    for i, layer in enumerate(mdm.seqTransDecoder.layers):
        handles.append(layer.register_forward_hook(layer_hook(i)))
    if ref.guided:
        def guided_hook(module, args, kwargs, output):
            t = int(args[1][0])
            if t in RECORD_STEPS:
                record[f"step{t}.x_t"] = _np(args[0])[:, :, 0, :].transpose(0, 2, 1)
                record[f"step{t}.guided_x0"] = _np(output)[:, :, 0, :].transpose(0, 2, 1)
        handles.append(ref.model.register_forward_hook(guided_hook, with_kwargs=True))
    else:
        def plain_pre(module, args, kwargs):
            t = int(args[1][0])
            if t in RECORD_STEPS:
                record[f"step{t}.x_t"] = _np(args[0])[:, :, 0, :].transpose(0, 2, 1)
        handles.append(mdm.register_forward_pre_hook(plain_pre, with_kwargs=True))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


def encode_text(ref: Reference, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Upstream ``bert_encode_text``: (hidden ``[B, L, 768]``, keep-mask ``[B, L]``), batch-first."""
    hidden, pad = ref.mdm.encode_text(list(texts))
    return _np(hidden).transpose(1, 0, 2), ~_np(pad).astype(bool)


def run(ref: Reference, texts: list[str], num_frames: int, seed: int, image_feature: np.ndarray | None = None,
        intermediates: bool = False) -> dict[str, np.ndarray]:
    """Generate like upstream ``generate.py::generate`` for one batch; returns NumPy arrays:
    ``sample`` (normalized ``[B, T, 407]``), ``features`` (de-normalized), ``joints [B, T, 34, 3]``,
    the text encoding, and (``intermediates=True``) the hooked tensors."""
    import torch

    import_upstream()
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from data_loaders.tensors import collate

    device, model = ref.device, ref.model
    n = len(texts)
    shape = (n, ref.mdm.njoints, ref.mdm.nfeats, num_frames)
    _, model_kwargs = collate([{"inp": torch.zeros(num_frames), "tokens": None, "lengths": num_frames, "text": t}
                               for t in texts])
    model_kwargs["y"] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in model_kwargs["y"].items()}
    if ref.args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(n, device=device) * ref.args.guidance_param
    model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])
    if ref.image_condition:
        if image_feature is None:
            raise ValueError("this reference is image-conditioned: pass image_feature")
        feature = np.asarray(image_feature, np.float32).reshape(-1, 768)
        feature = np.broadcast_to(feature, (n, 768))[:, None, :]  # upstream: (1, 768).repeat(n, 1, 1)
        model_kwargs["y"]["feature"] = torch.as_tensor(np.ascontiguousarray(feature), device=device)
    elif image_feature is not None:
        raise ValueError("this reference is text-only: image_feature must be None")

    mean = np.load(checkpoint_dir(ref.name) / "Mean.npy")
    std = np.load(checkpoint_dir(ref.name) / "Std.npy")
    record: dict[str, np.ndarray] = {}
    source = NoiseSource(seed)
    hooks = capture(ref, record) if intermediates else contextlib.nullcontext()
    with torch.no_grad(), numpy_randn(source, device), hooks:
        sample = ref.diffusion.p_sample_loop(model, shape, clip_denoised=False, model_kwargs=model_kwargs,
                                             skip_timesteps=0, init_image=None, progress=False, dump_steps=None,
                                             noise=None, const_noise=False)
    expected = [shape] * (ref.diffusion.num_timesteps + 1)
    if source.shapes != expected:
        raise RuntimeError(f"upstream drew noise {source.shapes[:3]}... ({len(source.shapes)} draws), expected "
                           f"{len(expected)} draws of {shape}")
    normalized = _np(sample)[:, :, 0, :].transpose(0, 2, 1)
    # generate.py: (sample.cpu().permute(0, 2, 3, 1) * std + mean).float(), then recover_from_ric
    features = (sample.cpu().permute(0, 2, 3, 1) * std + mean).float()
    joints = recover_from_ric(features, 34)
    joints = joints.view(-1, *joints.shape[2:]).permute(0, 2, 3, 1)  # [B, 34, 3, T]
    hidden, keep = encode_text(ref, texts)
    out = {"sample": normalized, "features": features[:, 0].numpy(), "joints": joints.permute(0, 3, 1, 2).numpy(),
           "text.hidden": hidden, "text.keep": keep}
    out.update(record)
    return out
