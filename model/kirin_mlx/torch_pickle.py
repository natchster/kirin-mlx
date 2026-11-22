# SPDX-License-Identifier: MIT
"""Read a PyTorch ``torch.save`` state dict with NumPy only, without executing pickled code.

Kirin's checkpoints are ``model.pt`` files: a zip archive holding ``<name>/data.pkl`` (a pickle
of the state dict) and one raw little-endian buffer per tensor storage under ``<name>/data/``.
``pickle.load`` would run whatever callables the pickle names, so this reader uses an unpickler
whose ``find_class`` allows only the handful of names a plain state dict uses (``OrderedDict``,
``torch._utils._rebuild_tensor_v2`` and the typed storage classes) and refuses everything else.
Tensors come back as NumPy arrays, so no PyTorch install is needed.
"""

from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

# torch.<Name>Storage -> NumPy dtype of the raw buffer
_STORAGES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "BFloat16Storage": np.uint16,  # widened to float32 below (NumPy has no bfloat16)
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name
        self.dtype = np.dtype(_STORAGES[name])


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    buffer, bf16 = storage
    if len(size) == 0:
        array = buffer[storage_offset:storage_offset + 1].reshape(())
    else:
        itemsize = buffer.dtype.itemsize
        array = np.lib.stride_tricks.as_strided(buffer[storage_offset:], shape=tuple(size),
                                                strides=tuple(s * itemsize for s in stride))
    array = np.array(array)  # own, contiguous copy
    if bf16:
        array = (array.astype(np.uint32) << 16).view(np.float32)
    return array


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None):
    if not isinstance(data, np.ndarray):
        raise pickle.UnpicklingError("parameter data is not a tensor")
    return data


class _StateDictUnpickler(pickle.Unpickler):
    def __init__(self, stream, archive: zipfile.ZipFile, prefix: str):
        super().__init__(stream)
        self._archive = archive
        self._prefix = prefix
        self._storages: dict[str, tuple[np.ndarray, bool]] = {}

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if (module, name) == ("torch._utils", "_rebuild_parameter"):  # nn.Parameter(tensor): keep the tensor
            return _rebuild_parameter
        if module == "torch" and name in _STORAGES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: not part of a plain state dict")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        _, storage_type, key, _location, numel = pid
        if not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError(f"unsupported storage type {storage_type!r}")
        if key not in self._storages:
            raw = self._archive.read(f"{self._prefix}/data/{key}")
            buffer = np.frombuffer(raw, dtype=storage_type.dtype.newbyteorder("<"))
            if buffer.size < numel:
                raise pickle.UnpicklingError(f"storage {key} holds {buffer.size} elements, expected {numel}")
            self._storages[key] = (buffer, storage_type.name == "BFloat16Storage")
        return self._storages[key]


def load_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """``{name: ndarray}`` from a zip-format ``torch.save`` file of a state dict (or of a dict
    holding one under ``"model_avg"``/``"model"``, as Kirin's training checkpoints do)."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a zip-format torch.save file (legacy pickles are not supported).")
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise ValueError(f"{path}: expected one data.pkl, found {pickles}")
        prefix = pickles[0].removesuffix("/data.pkl")
        byteorder = f"{prefix}/byteorder"
        if byteorder in archive.namelist() and archive.read(byteorder).decode().strip() != "little":
            raise ValueError(f"{path}: big-endian checkpoints are not supported")
        with archive.open(pickles[0]) as stream:
            state = _StateDictUnpickler(stream, archive, prefix).load()
    for key in ("model_avg", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    if not isinstance(state, dict) or not all(isinstance(v, np.ndarray) for v in state.values()):
        raise ValueError(f"{path} does not hold a state dict of tensors")
    return dict(state)
