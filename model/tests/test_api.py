"""Public API behaviour that needs no checkpoints (a randomly initialized model)."""

from __future__ import annotations

import kirin_mlx as kirin
import mlx.core as mx
import numpy as np
import pytest
from kirin_mlx.config import SPECS, canonical_name, spec_from_args
from kirin_mlx.model import Kirin, resolve_dtype
from kirin_mlx.tokenizer import WordPieceTokenizer

VOCAB = "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "horse", "dog", "runs"]) + "\n"


@pytest.fixture(scope="module")
def random_model():
    model = Kirin(SPECS["kirin_text_image"], np.zeros(407), np.ones(407), WordPieceTokenizer(VOCAB))
    model.eval()
    mx.eval(model.parameters())
    return model


def test_names_and_aliases():
    assert canonical_name("image") == "kirin_text_image" and canonical_name("TEXT") == "kirin_text"
    with pytest.raises(ValueError, match="Unknown Kirin model"):
        canonical_name("soma")


def test_dtypes():
    assert resolve_dtype("bfloat16") == mx.bfloat16 and resolve_dtype(mx.float16) == mx.float16
    with pytest.raises(ValueError, match="quantize"):
        resolve_dtype("int8")


def test_generate_shapes(random_model):
    out = random_model.generate("a horse runs", num_samples=2, num_frames=8, seed=0)
    assert out["joints"].shape == (2, 8, 34, 3) and out["features"].shape == (2, 8, 407)
    assert out["texts"] == ["a horse runs"] * 2 and out["lengths"].tolist() == [8, 8]
    assert np.isfinite(out["joints"]).all()


def test_motion_length_follows_upstream(random_model):
    assert random_model.generate("a dog", motion_length=1.0, seed=0)["joints"].shape[1] == 20
    assert random_model.generate("a dog", motion_length=60.0, guidance=1, seed=0)["joints"].shape[1] == kirin.MAX_FRAMES


def test_image_feature_shapes(random_model):
    feature = np.zeros(768, np.float32)
    for f in (feature, feature[None], np.zeros((2, 768), np.float32)):
        out = random_model.generate("a dog", image_feature=f, num_samples=2, num_frames=4, seed=0)
        assert out["joints"].shape[0] == 2
    with pytest.raises(ValueError, match="image_feature has shape"):
        random_model.generate("a dog", image_feature=np.zeros((3, 768)), num_samples=2, num_frames=4)
    with pytest.raises(ValueError, match="image_feature has shape"):
        random_model.generate("a dog", image_feature=np.zeros(512), num_frames=4)


def test_prompt_list_and_image_only(random_model):
    out = random_model.generate(["a dog", "a horse"], num_frames=4, seed=0)
    assert out["texts"] == ["a dog", "a horse"]
    with pytest.raises(ValueError, match="num_samples=3 but 2 prompts"):
        random_model.generate(["a dog", "a horse"], num_samples=3, num_frames=4)
    assert random_model.generate(None, image_feature=np.zeros(768), num_frames=4, seed=0)["texts"] == [" "]


def test_seeds(random_model):
    a = random_model.generate("a dog", num_frames=6, seed=1)["features"]
    b = random_model.generate("a dog", num_frames=6, seed=1)["features"]
    c = random_model.generate("a dog", num_frames=6, seed=2)["features"]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_explicit_noise_is_validated(random_model):
    with pytest.raises(ValueError, match="noise has shape"):
        random_model.generate("a dog", num_frames=6, noise=np.zeros((50, 1, 6, 407), np.float32))


def test_bad_frame_counts(random_model):
    with pytest.raises(ValueError, match="num_frames"):
        random_model.generate("a dog", num_frames=0)
    with pytest.raises(ValueError, match="at least 1"):
        random_model.generate("a dog", num_samples=0, num_frames=4)


def test_text_model_rejects_images():
    model = Kirin(SPECS["kirin_text"], np.zeros(407), np.ones(407), WordPieceTokenizer(VOCAB))
    with pytest.raises(ValueError, match="not image-conditioned"):
        model.generate("a dog", image_feature=np.zeros(768), num_frames=4)


def test_compute_dtype_switch_drops_compiled_state(random_model):
    random_model.generate("a dog", num_frames=4, seed=0)
    assert random_model._compiled_step is not None and random_model._time_table is not None
    random_model.set_compute_dtype("bfloat16")
    assert random_model.dtype == mx.bfloat16 and random_model._compiled_step is None
    assert random_model.mdm.seqTransDecoder.layers[0].linear1.weight.dtype == mx.bfloat16
    assert random_model.text_encoder.transformer.layer[0].ffn.lin1.weight.dtype == mx.float32  # encoder stays FP32
    assert np.isfinite(random_model.generate("a dog", num_frames=4, seed=0)["joints"]).all()
    random_model.set_dtype("float32")
    with pytest.raises(TypeError):
        random_model.set_dtype("float32", predicate=lambda k: True)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No converted weight file"):
        kirin.load(tmp_path / "missing.safetensors")
    with pytest.raises(FileNotFoundError, match="kirin-mlx download"):
        kirin.load("kirin_text", tmp_path)


def test_args_json_is_checked(tmp_path):
    import json

    args = {"arch": "trans_dec", "dataset": "animal", "text_encoder_type": "bert", "emb_policy": "concat",
            "latent_dim": 512, "layers": 8, "diffusion_steps": 50, "noise_schedule": "cosine", "sigma_small": True,
            "image_condition": False, "cond_mask_prob": 0.2}
    path = tmp_path / "args.json"
    path.write_text(json.dumps(args))
    assert spec_from_args("kirin_text", path).name == "kirin_text"
    path.write_text(json.dumps({**args, "arch": "trans_enc"}))
    with pytest.raises(ValueError, match="arch"):
        spec_from_args("kirin_text", path)
    path.write_text(json.dumps({**args, "layers": 12}))
    with pytest.raises(ValueError, match="layers"):
        spec_from_args("kirin_text", path)
