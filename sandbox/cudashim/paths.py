# SPDX-License-Identifier: MIT
"""Where the sandbox keeps its (gitignored) material, in one place."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / "sandbox"
UPSTREAM = SANDBOX / "upstream"  # briannlongzhao/kirin at the pinned commit (fetch_assets.py)
MOTION_GENERATION = UPSTREAM / "motion_generation"
GOLDEN = SANDBOX / "golden"  # reference fixtures (make_golden.py)
CHECKPOINTS = ROOT / "checkpoints"  # kirin/<model>/ and distilbert-base-uncased/ (kirin_mlx.hub)
