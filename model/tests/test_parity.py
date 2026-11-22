"""Parity with the upstream Kirin PyTorch generator (golden fixtures from sandbox/scripts/make_golden.py).

Every stage is **teacher-forced** with the exact tensors the PyTorch run produced, so an error
localizes to one module: tokenizer, DistilBERT, condition memory, each decoder layer, the
denoiser output, classifier-free guidance, one sampler step. End to end, both sides then sample
from the same noise (rebuilt from the seed) and must agree after all 50 steps.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from helpers import ALL_CASES, GOLDEN, golden_request, load_golden, load_model, rel_err

pytestmark = pytest.mark.weights

HOOK_STEP = 49
TEACHER_STEPS = ((49, 48), (25, 24), (1, 0))


def _text_golden(kind):
    path = GOLDEN / "text" / f"{kind}.npz"
    if not path.is_file():
        pytest.skip(f"{path} missing (sandbox/setup.sh)")
    return dict(np.load(path))


def _condition(model, golden):
    """The MLX condition built from the golden DistilBERT output (teacher-forced)."""
    image = golden.get("in.image_feature")
    if image is not None:
        image = mx.array(np.broadcast_to(image, (len(golden["in.texts"]), 768)))
    return model.condition(mx.array(golden["text.hidden"]), mx.array(golden["text.keep"]), image,
                           float(golden["in.guidance"]))


def test_tokenizer_matches_hugging_face():
    from kirin_mlx.tokenizer import WordPieceTokenizer

    golden = _text_golden("tokenizer")
    model = load_model("kirin_text")
    assert isinstance(model.tokenizer, WordPieceTokenizer)
    for i, text in enumerate(golden["texts"]):
        expected = golden["ids"][golden["offsets"][i]:golden["offsets"][i + 1]].tolist()
        assert model.tokenizer.encode(str(text)) == expected, repr(str(text))


def test_tokenizer_batch_padding_matches():
    golden = _text_golden("distilbert")
    ids, keep = load_model("kirin_text").tokenizer([str(t) for t in golden["texts"]])
    np.testing.assert_array_equal(ids, golden["input_ids"])
    np.testing.assert_array_equal(keep, golden["keep"])


def test_distilbert_matches():
    golden = _text_golden("distilbert")
    hidden, keep = load_model("kirin_text").encode_text([str(t) for t in golden["texts"]])
    hidden, keep = np.array(hidden), np.array(keep)
    np.testing.assert_array_equal(keep, golden["keep"])
    real = golden["keep"]  # padded rows are masked downstream, as upstream
    assert rel_err(hidden[real], golden["hidden"][real]) < 5e-6


@pytest.mark.parametrize("name,case", ALL_CASES)
def test_text_encoding_of_case(name, case):
    golden = load_golden(name, case)
    hidden, keep = load_model(name).encode_text([str(t) for t in golden["in.texts"]])
    np.testing.assert_array_equal(np.array(keep), golden["text.keep"])
    real = golden["text.keep"]
    assert rel_err(np.array(hidden)[real], golden["text.hidden"][real]) < 5e-6


@pytest.mark.parametrize("name,case", ALL_CASES)
def test_memory_and_input(name, case):
    golden = load_golden(name, case)
    model = load_model(name)
    cond = _condition(model, golden)
    x_t = mx.array(golden[f"step{HOOK_STEP}.x_t"])
    batch = x_t.shape[0]
    time = mx.broadcast_to(model.time_tokens()[HOOK_STEP], (batch, model.spec.latent_dim))
    branches = [("cond", slice(0, batch))] + ([("uncond", slice(batch, 2 * batch))] if cond.guided else [])
    for branch, rows in branches:
        memory = mx.concatenate([cond.prefix[rows], time[:, None, :], cond.suffix[rows]], axis=1)
        assert rel_err(memory, golden[f"step{HOOK_STEP}.{branch}.memory"]) < 2e-6, branch
        if f"step{HOOK_STEP}.{branch}.memory_keep" in golden:
            np.testing.assert_array_equal(np.array(cond.mask[rows]), golden[f"step{HOOK_STEP}.{branch}.memory_keep"])
    h = model.mdm.input_process(x_t)
    assert rel_err(h, golden[f"step{HOOK_STEP}.cond.input_process"]) < 2e-6
    h = h + model.mdm._pe[: x_t.shape[1]]
    assert rel_err(h, golden[f"step{HOOK_STEP}.cond.decoder_in"]) < 2e-6


@pytest.mark.parametrize("name,case", ALL_CASES)
def test_decoder_layers_teacher_forced(name, case):
    golden = load_golden(name, case)
    model = load_model(name)
    cond = _condition(model, golden)
    batch = golden["in.texts"].shape[0]
    branches = ["cond"] + (["uncond"] if cond.guided else [])
    for b, branch in enumerate(branches):
        rows = slice(b * batch, (b + 1) * batch)
        memory = mx.array(golden[f"step{HOOK_STEP}.{branch}.memory"])
        mask = cond.mask[rows][:, None, None, :]
        h = mx.array(golden[f"step{HOOK_STEP}.{branch}.decoder_in"])
        for i, layer in enumerate(model.mdm.seqTransDecoder.layers):
            out = layer(h, memory, mask)
            ref = golden[f"step{HOOK_STEP}.{branch}.layer{i}"]
            assert rel_err(out, ref) < 1e-5, f"{branch} layer {i}"
            h = mx.array(ref)


@pytest.mark.parametrize("name,case", ALL_CASES)
def test_denoiser_and_guidance(name, case):
    """Network outputs (both CFG branches) and guided x̂_0 at every recorded timestep."""
    golden = load_golden(name, case)
    model = load_model(name)
    cond = _condition(model, golden)
    batch = golden["in.texts"].shape[0]
    for t in sorted({t for pair in TEACHER_STEPS for t in pair}):
        x_t = mx.array(golden[f"step{t}.x_t"])
        time = mx.broadcast_to(model.time_tokens()[t], (cond.prefix.shape[0], model.spec.latent_dim))
        x_in = mx.concatenate([x_t, x_t]) if cond.guided else x_t
        out = model.mdm(x_in, time, cond.prefix, cond.suffix, cond.mask)
        assert rel_err(out[:batch], golden[f"step{t}.cond.x0"]) < 2e-5, f"t={t} cond"
        if cond.guided:
            assert rel_err(out[batch:], golden[f"step{t}.uncond.x0"]) < 2e-5, f"t={t} uncond"
            assert rel_err(model.predict_x0(x_t, t, cond), golden[f"step{t}.guided_x0"]) < 5e-5, f"t={t} guided"


@pytest.mark.parametrize("name,case", ALL_CASES)
def test_sampler_steps_teacher_forced(name, case):
    """x_t -> x_{t-1} with upstream's noise: posterior mean, variance and the CFG combination."""
    golden = load_golden(name, case)
    model = load_model(name)
    cond = _condition(model, golden)
    noise = golden_request(golden)["noise"]
    steps = model.spec.diffusion_steps
    for t, t_next in TEACHER_STEPS:
        x_t = mx.array(golden[f"step{t}.x_t"])
        eps = mx.array(noise[steps - t])  # noise[0] = x_T, noise[k] = ε used by the k-th step (t = 50 - k)
        got = model.step(x_t, t, eps, cond)
        ref = golden[f"step{t_next}.x_t"] if t > 0 else golden["sample"]
        assert rel_err(got, ref) < 5e-5, f"step {t}->{t_next}"


@pytest.mark.parametrize("name,case", ALL_CASES)
def test_end_to_end(name, case):
    """Same noise, 50 free-running steps (errors compound through guidance), then features and joints."""
    golden = load_golden(name, case)
    out = load_model(name).generate(**golden_request(golden))
    assert out["joints"].shape == golden["joints"].shape
    assert rel_err(out["features"], golden["features"]) < 2e-3
    scale = max(np.abs(golden["joints"]).max(), 1.0)
    assert np.abs(out["joints"] - golden["joints"]).max() < 5e-4 * scale


def test_schedule_matches_upstream():
    """The float32 coefficients equal upstream's ``_extract_into_tensor`` values exactly."""
    pytest.importorskip("torch")
    reference = pytest.importorskip("cudashim.reference")
    if not reference.available():
        pytest.skip("needs the upstream checkout (sandbox/setup.sh)")
    from kirin_mlx.diffusion import make_schedule

    reference.import_upstream()
    import torch
    from diffusion import gaussian_diffusion as gd
    from diffusion.respace import SpacedDiffusion, space_timesteps

    diffusion = SpacedDiffusion(use_timesteps=space_timesteps(50, [50]), betas=gd.get_named_beta_schedule("cosine", 50),
                                model_mean_type=gd.ModelMeanType.START_X, model_var_type=gd.ModelVarType.FIXED_SMALL,
                                loss_type=gd.LossType.MSE)
    ours = make_schedule(50, "cosine")
    t = torch.arange(50)
    shape = (50,)
    extract = gd._extract_into_tensor
    np.testing.assert_array_equal(ours.coef_x0, extract(diffusion.posterior_mean_coef1, t, shape).numpy())
    np.testing.assert_array_equal(ours.coef_xt, extract(diffusion.posterior_mean_coef2, t, shape).numpy())
    log_var = gd._extract_into_tensor(diffusion.posterior_log_variance_clipped, t, shape)
    sigma = torch.exp(0.5 * log_var).numpy()
    sigma[0] = 0.0
    np.testing.assert_allclose(ours.sigma, sigma, rtol=2e-7, atol=0)
