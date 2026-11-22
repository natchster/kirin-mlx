"""The torch-free ``model.pt`` reader and strict weight assignment."""

from __future__ import annotations

import io
import pickle
import zipfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from kirin_mlx import torch_pickle
from kirin_mlx.weights import assign


def _zip_with_pickle(tmp_path, payload: bytes):
    path = tmp_path / "evil.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/byteorder", "little")
    return path


class _Exploit:
    def __reduce__(self):
        import os

        return os.system, ("echo pwned",)


def test_refuses_code_execution(tmp_path):
    path = _zip_with_pickle(tmp_path, pickle.dumps(_Exploit(), protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="refusing to load posix.system|refusing to load os.system"):
        torch_pickle.load_state_dict(path)


def test_refuses_legacy_pickles(tmp_path):
    path = tmp_path / "legacy.pt"
    path.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(ValueError, match="not a zip-format"):
        torch_pickle.load_state_dict(path)


def test_matches_torch_save(tmp_path):
    torch = pytest.importorskip("torch")
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    state = {
        "contiguous": torch.randn(3, 5),
        "transposed": base.t(),  # non-contiguous strides
        "offset_view": base[1:, 2:],  # storage offset into a shared storage
        "shared": base,
        "half": torch.randn(4).half(),
        "bf16": torch.randn(4).bfloat16(),
        "int64": torch.arange(5),
        "bool": torch.tensor([True, False]),
        "scalar": torch.tensor(3.5),
        "param": torch.nn.Parameter(torch.randn(2, 2)),
    }
    path = tmp_path / "state.pt"
    torch.save({"model": state}, path)  # Kirin's training checkpoints nest the state dict
    loaded = torch_pickle.load_state_dict(path)
    assert set(loaded) == set(state)
    for key, value in state.items():
        expected = value.detach().float().numpy() if value.dtype == torch.bfloat16 else value.detach().numpy()
        np.testing.assert_array_equal(loaded[key], expected, err_msg=key)
        assert loaded[key].shape == tuple(value.shape)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)


def test_assign_is_strict():
    module = _Tiny()
    good = {"linear.weight": mx.ones((2, 3)), "linear.bias": mx.zeros((2,))}
    assign(module, good, "tiny")
    np.testing.assert_array_equal(np.array(module.linear.weight), 1)
    with pytest.raises(KeyError, match="missing"):
        assign(module, {"linear.weight": mx.ones((2, 3))}, "tiny")
    with pytest.raises(KeyError, match="unexpected"):
        assign(module, {**good, "linear.extra": mx.ones(1)}, "tiny")
    with pytest.raises(ValueError, match="shape mismatch"):
        assign(module, {**good, "linear.weight": mx.ones((3, 2))}, "tiny")


def test_buffer_roundtrip_independent_of_zip_name(tmp_path):
    """The archive folder inside the zip is named after the file by torch; any name must work."""
    torch = pytest.importorskip("torch")
    buffer = io.BytesIO()
    torch.save({"w": torch.ones(2)}, buffer)
    path = tmp_path / "renamed.pt"
    path.write_bytes(buffer.getvalue())
    np.testing.assert_array_equal(torch_pickle.load_state_dict(path)["w"], [1.0, 1.0])
