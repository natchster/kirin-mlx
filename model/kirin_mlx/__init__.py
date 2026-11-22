# SPDX-License-Identifier: MIT
"""MLX port of the Kirin animal motion generator for Apple silicon.

    import kirin_mlx as kirin
    model = kirin.load("kirin_text_image", "checkpoints/kirin/kirin_text_image", dtype="bfloat16")
    out = model.generate("a horse rears up on its hind legs", num_samples=3, seed=0)
    out["joints"]                    # [3, 120, 34, 3] joint positions, 20 fps

``kirin.download()`` fetches and verifies the checkpoints (both models + DistilBERT).
"""

from importlib.metadata import PackageNotFoundError, version

from . import hub
from .config import FEATURE_DIM, FPS, MAX_FRAMES, MODELS, NUM_JOINTS, canonical_name, get_spec
from .hub import download
from .model import DTYPES, Kirin, convert, from_checkpoints, from_converted, load

try:
    __version__ = version("kirin-mlx")  # single source: model/pyproject.toml
except PackageNotFoundError:  # imported from a source tree that is not installed
    __version__ = "0+unknown"
__all__ = ["DTYPES", "FEATURE_DIM", "FPS", "MAX_FRAMES", "MODELS", "NUM_JOINTS", "Kirin", "canonical_name", "convert",
           "download", "from_checkpoints", "from_converted", "get_spec", "hub", "load"]
