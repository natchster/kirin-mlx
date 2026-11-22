"""Behaviour of the real models (needs the checkpoints: ``kirin-mlx download``)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from helpers import MODELS, fresh_model, load_model, rel_err
from kirin_mlx.examples import PROMPTS, image_feature, upstream_noise

pytestmark = pytest.mark.weights


def _image(name, batch=1):
    return image_feature(3, batch) if name == "kirin_text_image" else None


@pytest.mark.parametrize("name", MODELS)
def test_generate_is_finite_and_seeded(name):
    model = load_model(name)
    a = model.generate(PROMPTS[0], image_feature=_image(name), num_samples=2, num_frames=40, seed=7)
    b = model.generate(PROMPTS[0], image_feature=_image(name), num_samples=2, num_frames=40, seed=7)
    assert a["joints"].shape == (2, 40, 34, 3) and np.isfinite(a["joints"]).all()
    np.testing.assert_array_equal(a["joints"], b["joints"])
    assert not np.allclose(a["joints"][0], a["joints"][1])  # different noise per sample


@pytest.mark.parametrize("name", MODELS)
def test_batch_rows_are_independent(name):
    """Two prompts in one batch == each prompt alone with its slice of the noise (padding included)."""
    model = load_model(name)
    texts = [PROMPTS[2], PROMPTS[3]]  # different token counts -> padding in the batch
    noise = upstream_noise(11, 2, 24)
    image = _image(name, 2)
    both = model.generate(texts, image_feature=image, num_frames=24, noise=noise)["features"]
    for i in range(2):
        alone = model.generate(texts[i], image_feature=None if image is None else image[i:i + 1], num_frames=24,
                               noise=noise[:, i:i + 1])["features"]
        # A different batch shape changes GEMM tiling (FP32 reduction order); 50 guided steps
        # amplify that rounding to ~2e-4, still far below any visible difference.
        assert rel_err(both[i:i + 1], alone) < 1e-3


@pytest.mark.parametrize("name", MODELS)
def test_compiled_equals_eager(name):
    model = load_model(name)
    noise = upstream_noise(5, 1, 16)
    kwargs = dict(image_feature=_image(name), num_frames=16, noise=noise)
    compiled = model.generate(PROMPTS[1], **kwargs)["features"]
    eager = model.generate(PROMPTS[1], compiled=False, **kwargs)["features"]
    assert rel_err(compiled, eager) < 1e-5


def test_batched_guidance_equals_two_passes():
    """The single 2B forward of a guided step equals upstream's two separate network calls."""
    model = load_model("kirin_text_image")
    hidden, keep = model.encode_text([PROMPTS[0], PROMPTS[2]])
    image = mx.array(image_feature(4, 2))
    cond = model.condition(hidden, keep, image, guidance=2.5)
    x = mx.array(upstream_noise(1, 2, 20)[0])
    t = 30
    time = mx.broadcast_to(model.time_tokens()[t], (2, 512))
    prefix, suffix, mask = model.mdm.memory(hidden, keep, image)
    u_prefix, u_suffix, _ = model.mdm.memory(hidden, keep, image, uncond=True)
    out = model.mdm(x, time, prefix, suffix, mask)
    out_uncond = model.mdm(x, time, u_prefix, u_suffix, mask)
    assert rel_err(model.predict_x0(x, t, cond), out_uncond + 2.5 * (out - out_uncond)) < 1e-5


def test_duplicate_prompts_are_encoded_once_with_same_result():
    model = load_model("kirin_text")
    hidden, keep = model.encode_text([PROMPTS[0], PROMPTS[3], PROMPTS[0]])
    ref_hidden, ref_keep = model.encode_text([PROMPTS[0], PROMPTS[3]])
    np.testing.assert_array_equal(np.array(keep), np.array(ref_keep)[[0, 1, 0]])
    assert rel_err(hidden, np.array(ref_hidden)[[0, 1, 0]]) == 0.0


def test_time_tokens_equal_per_step_embedding():
    model = load_model("kirin_text")
    table = np.array(model.time_tokens())
    for t in (0, 17, 49):
        single = model.mdm.timestep_embedding(mx.array([t]))
        assert rel_err(single, table[t:t + 1]) < 1e-6


@pytest.mark.parametrize("dtype,tol", [("bfloat16", 3e-2), ("float16", 1e-2)])
def test_half_precision_denoiser_close_to_fp32(dtype, tol):
    """One guided network evaluation in FP16/BF16 against FP32 (end to end, sampling noise dominates)."""
    ref, low = load_model("kirin_text_image"), fresh_model("kirin_text_image", dtype)
    x = mx.array(upstream_noise(2, 2, 60)[0])
    results = []
    for model in (ref, low):
        hidden, keep = model.encode_text([PROMPTS[0]] * 2)
        cond = model.condition(hidden, keep, mx.array(image_feature(5, 2)))
        results.append(np.array(model.predict_x0(x, 25, cond)))
    assert rel_err(results[1], results[0]) < tol
    out = low.generate(PROMPTS[0], image_feature=image_feature(5), num_samples=2, num_frames=60, seed=0)
    assert np.isfinite(out["joints"]).all()


def test_int8_runs_and_stays_close():
    ref, q = load_model("kirin_text"), fresh_model("kirin_text").quantize(8)
    assert q.quantization == 8
    noise = upstream_noise(3, 1, 30)
    a = ref.generate(PROMPTS[1], num_frames=30, noise=noise)["features"]
    b = q.generate(PROMPTS[1], num_frames=30, noise=noise)["features"]
    assert np.isfinite(b).all() and rel_err(b, a) < 0.1
    with pytest.raises(RuntimeError, match="already quantized"):
        q.quantize(8)


def test_convert_roundtrip(tmp_path):
    import kirin_mlx as kirin
    from helpers import checkpoint_dir, have_checkpoints

    if not have_checkpoints("kirin_text_image"):
        pytest.skip("checkpoints missing")
    path = kirin.convert("kirin_text_image", checkpoint_dir("kirin_text_image"), tmp_path / "k.safetensors")
    converted = kirin.load(path)
    original = load_model("kirin_text_image")
    noise = upstream_noise(9, 1, 12)
    kwargs = dict(image_feature=image_feature(9), num_frames=12, noise=noise)
    np.testing.assert_array_equal(converted.generate(PROMPTS[0], **kwargs)["features"],
                                  original.generate(PROMPTS[0], **kwargs)["features"])
    assert converted.tokenizer.encode(PROMPTS[2]) == original.tokenizer.encode(PROMPTS[2])
