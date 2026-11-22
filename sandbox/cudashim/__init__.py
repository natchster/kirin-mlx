# SPDX-License-Identifier: MIT
"""cudashim: a CUDA-emulation sandbox for Apple silicon (M1-M5), set up for Kirin.

There is no way to execute NVIDIA's binaries (cuDNN, libtorch_cuda, xformers, PyTorch3D's CUDA
kernels) on macOS; what works is substituting each CUDA layer at its API boundary. Kirin's motion
generator needs only one of them:

=================  ==========================================  ===============================
CUDA layer          Substitute on Apple silicon                 Module
=================  ==========================================  ===============================
torch.cuda          PyTorch MPS (Metal) + CPU fallback          :mod:`cudashim.torch_cuda`
=================  ==========================================  ===============================

:mod:`cudashim.reference` runs the upstream Kirin PyTorch generator as the oracle for the MLX
port in ``model/``, and ``cudashim-run`` runs an upstream script (e.g. ``generate.py``)
unchanged with ``cuda`` mapped to the Apple GPU. The README records which other parts of Kirin
(video reconstruction, mesh animation) can and cannot run this way.
"""

from __future__ import annotations

__all__ = ["enable_all", "enable_torch_cuda"]


def enable_torch_cuda():
    """Redirect ``cuda`` devices and the ``torch.cuda`` API to MPS. Returns the target device."""
    from .torch_cuda import enable

    return enable()


def enable_all():
    return enable_torch_cuda()
