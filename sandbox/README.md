# sandbox/: CUDA emulation on Apple silicon, for Kirin

A reproducible environment that runs the CUDA-targeting upstream Kirin code
([briannlongzhao/kirin](https://github.com/briannlongzhao/kirin), pinned at `252a5ec`) on an
M-series Mac. It is also the **reference oracle** that the MLX port in `../model` is checked
against. It follows the approach of the [motionbricks-mlx](https://github.com/natchster/motionbricks-mlx/blob/main/sandbox/README.md)
sandbox: substitute each CUDA layer at its API boundary, and don't try to run NVIDIA binaries.

## The short answer from the research

**No tool can execute CUDA binaries on a Mac**: Apple silicon has no NVIDIA driver or GPU and
supports no eGPU. libtorch_cuda, xformers' CUDA kernels and PyTorch3D's CUDA extension can't be
translated. What works is substituting each layer where its API starts. Kirin has three stages,
and they differ a lot in how much of that they need:

| Kirin stage | CUDA it relies on | On Apple silicon | Status here |
|---|---|---|---|
| **Motion generation** (`motion_generation/`, the part with released weights) | `torch.cuda` only: `dist_util.dev()`, a `pipeline(device=0)` for DINOv3, and float64 diffusion tables moved to the GPU | PyTorch **MPS** behind a `torch.cuda` shim; float64 → float32 on the GPU | **runs unchanged**, tested (`cudashim-run generate.py`, reference on MPS) |
| Video reconstruction (`run_reconstruction.py`: AniMer, SpaTrackerV2, SAM 3, DINOv3) | `.cuda()` calls; **xformers** `memory_efficient_attention`/`SwiGLU` (with PyTorch fallbacks in the vendored DINOv2 layers); **PyTorch3D** built against CUDA 12.4; pycolmap/pyceres; `torch==2.4.1+cu124` pinned because "the conda build gives different tracker results" | MPS for the PyTorch parts. xformers has no macOS build, so the fallback paths are used. PyTorch3D builds CPU-only on macOS. SAM 3 and DINOv3 are gated on Hugging Face | not set up: needs gated weights and a CPU PyTorch3D build, and its tracker results are documented as build-sensitive |
| Mesh animation (`animate_mesh.py`) | PyTorch3D rendering/fitting on CUDA; the **Rodin** web API (Hyper3D) for the mesh | PyTorch3D CPU; Rodin is a remote paid API (`RODIN_API_KEY`) | not set up: an external service plus CPU PyTorch3D |

**For now, the MLX port and this sandbox cover only motion generation**, the part users run for
inference (text and/or image → animal motion). Reconstruction and mesh animation are surveyed above
but not ported or set up. Survey method: `grep` of the pinned tree and both submodules (AniMer
`4d8be3b`, SpaTrackerV2 `da6a92d`) for `.cuda()`, `torch.cuda` and native or CUDA-only imports.
xformers appears in 7 SpaTrackerV2 files, PyTorch3D in 8 files, and there is one custom CUDA
`setup.py` (AniMer).

| CUDA layer used by the generator | Substitute | Fidelity | Module |
|---|---|---|---|
| `torch.cuda` / `.cuda()` / `device="cuda"` | PyTorch **MPS** (Metal), CPU fallback for missing ops | same kernels' math. FP64 is stored as FP32 on the GPU, which is exact for upstream's "move, index, `.float()`" use | `cudashim/torch_cuda.py` |
| `from_pretrained("distilbert/distilbert-base-uncased")` at run time (unpinned) | the pinned, SHA-256-verified local copy | same weights | `cudashim/run.py`, `cudashim/reference.py` |
| (reference) the upstream PyTorch generator | upstream `motion_generation/`, pinned commit | it *is* the reference | `cudashim/reference.py` |

The raw-CUDA C++ layer (CuMetal) from the motionbricks sandbox isn't included: none of the
generator's code is CUDA C++.

Two findings came out of building this:

- **guided-diffusion moves float64 tables to the GPU.** Its `_extract_into_tensor` runs
  `th.from_numpy(float64_array).to(device)[t].float()`. That works on CUDA and fails on MPS
  ("MPS framework doesn't support float64"), so the upstream sampler doesn't run on a Mac even with
  a device remap. The shim stores float64 tensors sent to the cuda alias as float32, with a
  one-time warning. Rounding to float32 commutes with the indexing that comes before `.float()`, so
  the result is bit-identical. `test_torch_cuda_shim.py` checks this.
- **torch._dynamo indexes torch functions by identity.** The motionbricks shim assigned
  `torch.mps.manual_seed` itself to `torch.cuda.manual_seed`. Once transformers imports
  `torch._dynamo`, its trace-rule table finds one object under two names with different rules,
  and the import fails. The patches are now fresh wrappers, with a regression test.

## Setup

The sandbox is only needed for the reference oracle, the golden fixtures, the parity tests and
the cross-backend benchmarks. The MLX model installs and runs without it.

```sh
sandbox/setup.sh                 # env, upstream @ 252a5ec, checkpoints (~500 MB), golden fixtures (~25 s)
sandbox/setup.sh --aim           # also: real DINOv3 features + captions of 8 AiM3D test clips
sandbox/setup.sh --force         # regenerate every fixture
uv run --all-packages cudashim-doctor
```

`setup.sh` is safe to rerun. It does the following:

1. `uv sync --all-packages`: Python 3.12, MLX, PyTorch 2.15 (MPS), **transformers 5.0.0** (the
   version upstream pins), and the rest of upstream `generate.py`'s import chain (moviepy 1.0.3,
   spaCy 3.8, matplotlib, scikit-learn, SciPy, joblib), so the unmodified script can run.
2. Clones briannlongzhao/kirin at the pinned commit into `sandbox/upstream/`. With `--submodules`,
   `fetch_assets.py` also checks out AniMer and SpaTrackerV2 for reading; the generator doesn't use them.
3. Downloads both generators and DistilBERT with the model package's own downloader
   (`kirin_mlx.hub`, the same one `kirin-mlx download` uses). Each file is pinned to a Hugging Face
   revision and checked against its size and SHA-256.
4. Records golden fixtures with the upstream PyTorch generator on the CPU (FP32):
   - `sandbox/golden/text/`: the Hugging Face tokenizer on 15 tricky strings (accents, CJK,
     control characters, literal special tokens, Greek final sigma, a 120-character word), and
     DistilBERT hidden states for a padded batch.
   - `sandbox/golden/<model>/<case>.npz`, 8 cases. Text + image; image only (upstream `--no_text`);
     text only on the image model (`--no_image`); 196 frames; per-sample images; the text model with
     repeated and mixed prompts; guidance off.

   Each case stores the inputs, the text encoding, the denoiser's per-layer tensors at t = 49 for
   both guidance branches, x_t and the network outputs at t ∈ {49, 48, 25, 24, 1, 0}, and the
   final sample, features and joints. `meta.json` records the device and versions.
5. With `--aim`, fetches real DINOv3 image features, which the image model needs. DINOv3 itself is
   gated, but upstream's AiM3D test archive (`AiM_preview.zip`, 14.5 GB) ships each clip's
   `mean_feature_dinov3.npy`. `fetch_aim_samples.py` reads only the zip's central directory over
   HTTP range requests (~15 MB, cached), then the few members it needs, checking each one's CRC-32.

**How the noise is matched.** The MLX port can't reproduce PyTorch's RNG, so the oracle replaces
`torch.randn`/`torch.randn_like` during sampling with draws from a NumPy generator
(`kirin_mlx.examples.NoiseSource`). The draws follow upstream's order and layout: x_T, then one ε
per step, each `[B, 407, 1, T]`. The oracle asserts exactly 51 draws of the expected shape.
`kirin_mlx.examples.upstream_noise(seed, B, T)` rebuilds the same sequence in the MLX layout, so
both sides sample from identical noise.

## Usage

```python
import cudashim
cudashim.enable_all()            # cuda -> mps for torch

import torch
x = torch.randn(1024, 1024, device="cuda")   # lands on mps:0
model = MyNet().cuda()                         # NVIDIA-style code, unchanged
```

Run an upstream script unchanged, with `cuda` on the Apple GPU and DistilBERT read from the
verified local copy (no network):

```sh
uv run --all-packages cudashim-run sandbox/upstream/motion_generation/generate.py \
    --model_path checkpoints/kirin/kirin_text/model.pt --text_prompt "a dog runs forward and then jumps" \
    --num_samples 2 --num_repetitions 1 --output_dir out/dog
uv run --all-packages cudashim-run sandbox/upstream/motion_generation/generate.py \
    --model_path checkpoints/kirin/kirin_text_image/model.pt --text_prompt "a brown bear walks slowly" \
    --image_feature_path sandbox/assets/aim/bear/2AyBtNUWgg0_109_001/mean_feature_dinov3.npy --output_dir out/bear
```

`--cpu` leaves CUDA unavailable, so upstream falls back to the CPU. `--hf-local REPO=DIR` adds more
redirects. Point `facebook/dinov3-vitb16-pretrain-lvd1689m` at a local copy if you have access, and
`--image_path` works too.

## What is verified (`uv run --all-packages pytest sandbox/tests`)

`KIRIN_REQUIRE_FIXTURES=1` turns every skip into a failure.

| Test | Proves |
|---|---|
| `test_torch_cuda_shim.py` | `device="cuda"`, `.cuda()`, events, sync and autocast land on MPS and match CPU numerics. Float64 sent to the alias becomes float32, exactly for the move/index/`.float()` pattern. `import torch._dynamo` (and transformers) works with the shim on. |
| `test_reference_on_mps.py` | **The upstream generator, moved with `.cuda()`**, runs on the Apple GPU and matches the CPU golden fixtures with the same noise: features within 5e-3 relative, joints within 2e-3 of the motion's extent. |
| `test_upstream_cli.py` | **Upstream's own `generate.py`, unmodified**, runs end to end through `cudashim-run` on MPS and on the CPU with `HF_HUB_OFFLINE=1`, and writes its usual `sample_NN.npy` and `results_*.npy`. |

## Limits

- **No CUDA graphs, streams or custom CUDA kernels.** The generator needs none of them.
  Reconstruction (xformers, PyTorch3D) and mesh animation (PyTorch3D, Rodin API) aren't set up here;
  see the table above.
- **FP64 on the GPU becomes FP32.** This is exact for upstream's use. Code that really needs FP64
  arithmetic on the device should run with `--cpu`.
- **MPS is not the FP32 reference.** Its kernels round differently, and the differences grow over
  50 guided steps. Only the CPU fixtures are the reference; `make_golden.py --device mps` writes to
  `sandbox/golden-mps/`.
