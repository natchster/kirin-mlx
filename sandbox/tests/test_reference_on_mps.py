"""The upstream Kirin generator, moved with ``.cuda()`` as on an NVIDIA box, runs on the Apple GPU
through the shim and agrees with the CPU golden fixtures (same noise)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cudashim import reference, torch_cuda  # noqa: E402
from cudashim.paths import GOLDEN  # noqa: E402

pytestmark = [
    pytest.mark.reference,
    pytest.mark.skipif(not reference.available(), reason="run sandbox/setup.sh"),
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS"),
]


@pytest.mark.parametrize("name,case", [("kirin_text_image", "text_image"), ("kirin_text", "mixed")])
def test_upstream_generator_on_cuda_alias(name, case):
    path = GOLDEN / name / f"{case}.npz"
    if not path.is_file():
        pytest.skip("needs golden fixtures (sandbox/setup.sh)")
    meta = path.with_name("meta.json")
    assert json.loads(meta.read_text())["device"] == "cpu", "MPS vs MPS would prove nothing"
    golden = dict(np.load(path))
    texts = [str(t) for t in golden["in.texts"]]
    image = golden.get("in.image_feature")
    with torch_cuda.enabled():
        ref = reference.build(name, "cuda", image=image is not None, guidance=float(golden["in.guidance"]))
        assert next(ref.model.parameters()).device.type == "mps"
        out = reference.run(ref, texts, int(golden["in.num_frames"]), int(golden["in.seed"]), image)
    scale = max(np.abs(golden["joints"]).max(), 1.0)
    assert np.abs(out["features"] - golden["features"]).max() < 5e-3 * np.abs(golden["features"]).max()
    assert np.abs(out["joints"] - golden["joints"]).max() < 2e-3 * scale
