"""Upstream's own ``motion_generation/generate.py``, unmodified, runs end to end on the Mac through
``cudashim-run`` (cuda -> MPS, DistilBERT from the verified local copy, no network)."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cudashim import reference  # noqa: E402
from cudashim.paths import CHECKPOINTS, MOTION_GENERATION  # noqa: E402

pytestmark = [
    pytest.mark.reference,
    pytest.mark.skipif(not reference.available(), reason="run sandbox/setup.sh"),
]


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_generate_py_runs_unchanged(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    model_pt = CHECKPOINTS / "kirin" / "kirin_text" / "model.pt"
    if not model_pt.is_file():
        pytest.skip("checkpoints missing (kirin-mlx download)")
    out = tmp_path / "dog"
    cmd = [sys.executable, "-m", "cudashim.run", *(["--cpu"] if device == "cpu" else []),
           str(MOTION_GENERATION / "generate.py"), "--model_path", str(model_pt),
           "--text_prompt", "a dog runs forward and then jumps", "--num_samples", "2", "--num_repetitions", "1",
           "--motion_length", "1.0", "--seed", "3", "--output_dir", str(out)]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}  # proves the run needs no download
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    if device == "mps":
        assert "cudashim: cuda -> mps" in result.stderr
    samples = sorted(out.glob("sample_*.npy"))
    assert len(samples) == 2
    for sample in samples:
        joints = np.load(sample)
        assert joints.shape == (20, 34, 3) and np.isfinite(joints).all()
    results = np.load(out / "results_kirin_text.npy", allow_pickle=True).item()  # upstream's own output format
    assert results["motions"].shape == (2, 20, 34, 3)


def test_generate_py_image_feature_on_mps(tmp_path):
    """The image model with --image_feature_path (a float64 .npy, like AiM3D's mean_feature_dinov3.npy)."""
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    model_pt = CHECKPOINTS / "kirin" / "kirin_text_image" / "model.pt"
    if not model_pt.is_file():
        pytest.skip("checkpoints missing (kirin-mlx download)")
    from kirin_mlx.examples import image_feature

    feature = tmp_path / "mean_feature_dinov3.npy"
    np.save(feature, image_feature(0).astype(np.float64))
    out = tmp_path / "bear"
    cmd = [sys.executable, "-m", "cudashim.run", str(MOTION_GENERATION / "generate.py"), "--model_path", str(model_pt),
           "--text_prompt", "a brown bear walks slowly", "--image_feature_path", str(feature), "--num_samples", "1",
           "--num_repetitions", "1", "--motion_length", "1.0", "--seed", "1", "--output_dir", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert result.returncode == 0, result.stderr[-3000:]
    joints = np.load(out / "sample_00.npy")
    assert joints.shape == (20, 34, 3) and np.isfinite(joints).all()
