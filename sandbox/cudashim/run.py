# SPDX-License-Identifier: MIT
"""``cudashim-run``: run an upstream CUDA-targeting script unchanged on the Apple GPU.

    uv run --all-packages cudashim-run sandbox/upstream/motion_generation/generate.py \\
        --model_path checkpoints/kirin/kirin_text/model.pt --text_prompt "a dog runs forward" \\
        --output_dir out/dog --num_samples 2 --num_repetitions 1

The script runs as ``__main__`` from its own directory (upstream scripts use relative paths and
top-level imports such as ``from utils...``), with:

* ``cuda`` mapped to MPS through :mod:`cudashim.torch_cuda` (``--cpu`` leaves CUDA unavailable,
  so upstream's ``dist_util.dev()`` falls back to the CPU);
* Hugging Face repositories redirected to local, verified folders (``--hf-local REPO=DIR``; by
  default ``distilbert/distilbert-base-uncased`` -> ``checkpoints/distilbert-base-uncased``), so
  the run uses the pinned weights and needs no network.

Paths given on the command line are resolved against the current directory before the switch.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from .paths import CHECKPOINTS

DEFAULT_HF_LOCAL = {"distilbert/distilbert-base-uncased": str(CHECKPOINTS / "distilbert-base-uncased")}


def redirect_hf(mapping: dict[str, str]) -> None:
    """Make ``from_pretrained(repo)`` (models, tokenizers, configs, pipelines) load ``mapping[repo]``."""
    import transformers

    def wrap(cls):
        original = cls.from_pretrained.__func__

        def from_pretrained(klass, name, *args, **kwargs):
            return original(klass, mapping.get(str(name), name), *args, **kwargs)

        cls.from_pretrained = classmethod(from_pretrained)

    for cls_name in ("AutoTokenizer", "AutoModel", "AutoConfig", "AutoImageProcessor", "AutoProcessor"):
        wrap(getattr(transformers, cls_name))
    original_pipeline = transformers.pipeline

    def pipeline(*args, model=None, **kwargs):
        return original_pipeline(*args, model=mapping.get(str(model), model), **kwargs)

    transformers.pipeline = pipeline


def _absolute(arg: str) -> str:
    """Resolve an existing relative path argument against the caller's directory."""
    path = Path(arg)
    if arg.startswith("-") or path.is_absolute():
        return arg
    if path.exists() or ("/" in arg and path.parent.exists()):  # existing input, or output in an existing dir
        return str(path.resolve())
    return arg  # e.g. a text prompt


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cudashim-run", description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="do not map cuda to MPS (CUDA stays unavailable)")
    parser.add_argument("--hf-local", action="append", default=[], metavar="REPO=DIR",
                        help="load a Hugging Face repository from a local folder (repeatable)")
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)

    mapping = dict(DEFAULT_HF_LOCAL)
    for item in ns.hf_local:
        repo, _, folder = item.partition("=")
        mapping[repo] = str(Path(folder).resolve())
    script = Path(ns.script).resolve()
    args = [_absolute(a) for a in ns.args]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not ns.cpu:
        from .torch_cuda import enable

        print(f"cudashim: cuda -> {enable()}", file=sys.stderr)
    redirect_hf(mapping)
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
