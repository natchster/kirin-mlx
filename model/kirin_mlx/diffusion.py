# SPDX-License-Identifier: MIT
# Port of the sampling path of guided-diffusion (OpenAI, MIT) as used by MDM/Kirin
# (motion_generation/diffusion/gaussian_diffusion.py and respace.py).
"""The ancestral sampler Kirin generates with: ``p_sample_loop`` over 50 cosine-schedule steps.

Released configuration: ``predict_xstart=True`` (the network outputs x̂_0), ``sigma_small=True``
(``FIXED_SMALL`` variance = the true posterior variance), ``clip_denoised=False``, no timestep
rescaling, and a ``SpacedDiffusion`` that keeps all 50 steps. Each step is

    x̂_0      = uncond + s · (cond − uncond)                 (classifier-free guidance, s = 2.5)
    μ         = c1[t] · x̂_0 + c2[t] · x_t                    (posterior mean q(x_{t-1} | x_t, x̂_0))
    x_{t-1}   = μ + σ[t] · ε,   σ[t] = exp(½ log σ²[t]),  σ[0] = 0

The schedule is computed exactly as upstream: float64 NumPy, including ``SpacedDiffusion``'s
round trip through ``alphas_cumprod``, then cast to float32 per coefficient, which is what
``_extract_into_tensor`` hands the float32 tensors. The step arithmetic runs in float32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def cosine_betas(num_steps: int, max_beta: float = 0.999) -> np.ndarray:
    """``betas_for_alpha_bar`` with the cosine ᾱ(t) of Nichol & Dhariwal (float64)."""

    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    return np.array([min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
                     for i in range(num_steps)])


def linear_betas(num_steps: int) -> np.ndarray:
    scale = 1000 / num_steps
    return np.linspace(scale * 0.0001, scale * 0.02, num_steps, dtype=np.float64)


@dataclass(frozen=True)
class Schedule:
    """Per-step float32 coefficients, indexed by the diffusion timestep ``t`` (0 = last step)."""

    coef_x0: np.ndarray  # posterior_mean_coef1
    coef_xt: np.ndarray  # posterior_mean_coef2
    sigma: np.ndarray  # exp(0.5 * posterior_log_variance_clipped), 0 at t = 0 (nonzero_mask)
    alphas_cumprod: np.ndarray  # float64, for reference / q_sample

    @property
    def num_steps(self) -> int:
        return len(self.coef_x0)


def _base_arrays(betas: np.ndarray) -> dict[str, np.ndarray]:
    """``GaussianDiffusion.__init__`` (float64)."""
    betas = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "alphas_cumprod": alphas_cumprod,
        "posterior_log_variance_clipped": np.log(np.append(posterior_variance[1], posterior_variance[1:])),
        "posterior_mean_coef1": betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
    }


def make_schedule(num_steps: int = 50, noise_schedule: str = "cosine") -> Schedule:
    if noise_schedule == "cosine":
        betas = cosine_betas(num_steps)
    elif noise_schedule == "linear":
        betas = linear_betas(num_steps)
    else:
        raise ValueError(f"unknown noise schedule {noise_schedule!r}")
    # SpacedDiffusion(use_timesteps=all steps): betas re-derived from the base alphas_cumprod.
    base = _base_arrays(betas)
    spaced, last = [], 1.0
    for alpha_cumprod in base["alphas_cumprod"]:
        spaced.append(1 - alpha_cumprod / last)
        last = alpha_cumprod
    arrays = _base_arrays(np.array(spaced))
    log_var = arrays["posterior_log_variance_clipped"].astype(np.float32)
    sigma = np.exp(np.float32(0.5) * log_var).astype(np.float32)
    sigma[0] = 0.0  # upstream: nonzero_mask = (t != 0)
    return Schedule(arrays["posterior_mean_coef1"].astype(np.float32),
                    arrays["posterior_mean_coef2"].astype(np.float32), sigma, arrays["alphas_cumprod"])
