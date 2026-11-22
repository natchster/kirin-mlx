# model/: the Kirin motion generator on MLX

`kirin_mlx` ports the motion generator of [Kirin](https://github.com/briannlongzhao/kirin)
(ECCV 2026) to [MLX](https://github.com/ml-explore/mlx), for both released models. It runs on the
Apple GPU through Metal and depends only on `mlx` and `numpy`.

- **`kirin_text_image`**: text, a DINOv3 image feature of the animal, or both.
- **`kirin_text`**: text only.

> [!IMPORTANT]
> **Scope: the motion generator only, for now.** Kirin's video reconstruction (`run_reconstruction.py`),
> mesh animation (`animate_mesh.py`) and the DINOv3 image encoder are not part of this package. The
> image model takes a precomputed DINOv3 feature, not an image file. See [What is ported](#what-is-ported).

## What is ported

Everything upstream's `motion_generation/generate.py` runs to turn a prompt into joint motion:

```
text ──WordPiece──► DistilBERT (6 layers, 768-d) ──embed_text──┐
DINOv3 [CLS] feature (768) ──embed_image──────────────────────┤ memory = [image? | timestep | text tokens]
diffusion step t ──sinusoidal PE──► MLP ──────────────────────┘         │ cross-attention
x_t [N, T, 407] ──poseEmbedding + PE──► 8 × post-norm TransformerDecoderLayer (512-d, 4 heads) ──poseFinal──► x̂_0
   × 50 steps: classifier-free guidance (s = 2.5) + posterior-mean update (cosine schedule, σ² = posterior variance)
──► de-normalize ──► integrate root yaw/velocity, rotate local joints ──► joints [N, T, 34, 3] (20 fps)
```

| | `kirin_text_image` | `kirin_text` |
|---|---|---|
| inputs | prompt(s), optional DINOv3 ViT-B/16 [CLS] feature `[768]` | prompt(s) |
| outputs | `joints [N, T, 34, 3]`, `features [N, T, 407]` | same |
| denoiser parameters | 27.0 M (+ DistilBERT 66 M, run once per call) | 26.6 M (+ DistilBERT) |
| motion length | T = min(196, 20 × seconds), as upstream | same |

**What is not ported (yet).** The DINOv3 image encoder is gated on Hugging Face. As upstream does with
`--image_feature_path`, the image model takes the 768-d feature directly. Real features of AiM3D
test clips come from `sandbox/scripts/fetch_aim_samples.py`. Kirin's video reconstruction and
mesh animation stages are not ported either; [sandbox/README.md](../sandbox/README.md) explains why.

## Usage

```python
import kirin_mlx as kirin
import numpy as np

model = kirin.load("kirin_text_image", "checkpoints/kirin/kirin_text_image", dtype="bfloat16")
feature = np.load("mean_feature_dinov3.npy")                      # [1, 768] DINOv3 [CLS] feature
out = model.generate("a horse rears up on its hind legs", image_feature=feature, num_samples=3, seed=0)
out["joints"]                                                      # [3, 120, 34, 3], 6 s at 20 fps

model.generate(None, image_feature=feature)                        # image only (upstream --no_text)
model.generate("a dog runs forward", motion_length=9.8)            # text only on the image model (--no_image)
model.generate(["a cat jumps", "a deer walks"], guidance=1)        # one prompt per sample, no guidance

kirin.convert("kirin_text", "checkpoints/kirin/kirin_text", "weights/kirin_text.safetensors")  # one file
text_model = kirin.load("weights/kirin_text.safetensors").quantize(8)                         # experimental INT8
```

`generate` checks its inputs and raises `ValueError` on a wrong shape or an image passed to the
text model. Model names are `kirin_text_image` (alias `image`) and `kirin_text` (alias `text`).
The random numbers come from MLX, so a seed is reproducible within MLX but doesn't reproduce a
PyTorch run. To replay one, pass its noise: `noise=[x_T, ε_49, …, ε_0]`, shaped `[51, N, T, 407]`.

Install and get the weights (MLX + NumPy only, no sandbox):

```sh
uv sync --package kirin-mlx                       # from the repository root; or: pip install ./model
uv run --package kirin-mlx kirin-mlx download     # checkpoints/{kirin/*, distilbert-base-uncased}, SHA-256 verified
```

From the command line (output files like upstream: `sample_NN.npy` `[T, 34, 3]` + `results_<model>.npy`):

```sh
uv run --package kirin-mlx kirin-mlx generate --model kirin_text --text "a dog runs forward and then jumps" --num_samples 3
uv run --package kirin-mlx kirin-mlx generate --model image --image-feature mean_feature_dinov3.npy --no_text --dtype bfloat16
uv run --package kirin-mlx kirin-mlx bench --model kirin_text_image --dtype bfloat16 --batch 1 4 16
uv run --package kirin-mlx kirin-mlx convert --model kirin_text --output weights/kirin_text.safetensors
```

`download` fetches `briannlongzhao/kirin` at revision `c3a9114` and
`distilbert/distilbert-base-uncased` at `12040ac` over plain HTTPS. Every file is checked against
the size and SHA-256 in `assets/checkpoints.sha256`, and files that are already verified are skipped.
Downloads stream into a unique `.part` file that is renamed into place only after verification.

## Design notes

- **No PyTorch at load time, and no code execution.** `model.pt` is a pickle, and `pickle.load`
  runs any callable the pickle names. `torch_pickle.py` uses an unpickler whose `find_class`
  allows only what a plain state dict uses (`OrderedDict`, `_rebuild_tensor_v2`,
  `_rebuild_parameter`, the typed storages) and refuses everything else. It reads tensor storages
  from the zip with NumPy, and its output equals `torch.load` bit for bit.
- **A plain-Python tokenizer.** `tokenizer.py` re-implements `tokenizer.json`'s pipeline:
  literal special tokens, BertNormalizer, BertPreTokenizer, WordPiece, padding. It matches the Hugging
  Face tokenizer on every test string, including one subtle point: the Rust normalizer lowercases
  per character, so word-final `Σ` becomes `σ`, while Python's `str.lower()` would give `ς`.
- **Batch-first layout.** Upstream runs `nn.Transformer*` sequence-first and permutes at every
  boundary. The port uses `(batch, length, channels)` throughout. PyTorch's packed
  `in_proj_weight` stays packed for self-attention (one GEMM for q, k, v) and is split into
  `q_proj` / `kv_proj` for cross-attention, which is the split PyTorch itself makes.
- **Guidance as one batch.** Upstream's `ClassifierFreeSampleModel` calls the network twice per
  step. Here the conditional and unconditional rows form one `2N` batch. Building the
  unconditional memory follows upstream's `mask_cond` exactly: it zeroes the DistilBERT features
  *before* `embed_text`, so every text token becomes `embed_text.bias`, and zeroes the image token
  *after* `embed_image`. The padding mask is unchanged.
- **Constant work is hoisted.** The timestep tokens of all 50 steps are computed once per model.
  DistilBERT runs once per call, and only on unique prompts. Each denoising step is one
  `mx.compile`-d graph, and the 50 steps are queued lazily and evaluated together.
- **Exact schedule.** `diffusion.py` computes the cosine schedule the way upstream does: float64,
  including `SpacedDiffusion`'s round trip through `alphas_cumprod`. A test checks that its float32
  coefficients equal upstream's `_extract_into_tensor` values exactly.
- **Mixed precision only where it is safe.** `float16`/`bfloat16` apply to the denoiser. The
  text encoder, the diffusion update, de-normalization and joint recovery stay FP32.
- **Generation is compute-bound.** Each sample is T = 120 frame tokens, and guidance doubles
  that, so a B = 1 generation is about 540 GFLOP. Smaller weights help less than with MotionBricks;
  see [benchmarks/](benchmarks/README.md).

## Verification

`uv run --package kirin-mlx --extra test pytest model/tests` (or `pip install "./model[test]" &&
pytest model/tests`) runs the suite without the sandbox. Without torch, only 3 tests skip.

| Tests | Need | Without them |
|---|---|---|
| weight-free unit tests: decoder layer and DistilBERT block against NumPy, schedule, joint recovery, `mask_cond` semantics, tokenizer, `.pt` reader security, downloader verification, API checks (`test_layers.py`, `test_tokenizer.py`, `test_weights.py`, `test_hub.py`, `test_api.py`) | nothing | always run |
| model behaviour (`test_model.py`) | the checkpoints | skipped |
| parity with upstream (`test_parity.py`) | the golden fixtures (`sandbox/setup.sh`) | skipped |

`KIRIN_CHECKPOINTS` and `KIRIN_GOLDEN` point the tests at other locations, and
`KIRIN_REQUIRE_FIXTURES=1` turns every skip into a failure.

The parity tests compare against Kirin's own PyTorch generator, using fixtures recorded by
`sandbox/scripts/make_golden.py` over 8 request families. Every stage is **teacher-forced** with
the exact tensors the PyTorch run produced, so an error localizes to one module. End to end, both
sides sample from identical noise (see [sandbox/README.md](../sandbox/README.md#setup)).

| Stage (8 cases × both guidance branches) | Max relative error vs PyTorch FP32 |
|---|---|
| Tokenizer (15 tricky strings + padded batch) | **identical ids and masks** |
| DistilBERT hidden states (real tokens) | ≤ 1.7e-6 |
| Condition memory, input projection | ≤ 2e-6 |
| Each of the 8 decoder layers | ≤ 1.3e-6 |
| Denoiser output x̂_0 / guided x̂_0 (t ∈ {49, 48, 25, 24, 1, 0}) | ≤ 8.4e-6 / ≤ 8.4e-6 |
| One sampler step x_t → x_{t-1} | ≤ 4.1e-6 |
| Diffusion coefficients | **bit-identical** (posterior mean), ≤ 2e-7 (σ) |
| End to end after 50 free-running guided steps: features / joints | ≤ 7.4e-4 / ≤ 3.3e-4 of the motion's extent |

The end-to-end error is the per-step FP32 rounding (~4e-6), compounded over 50 steps of guidance
with scale 2.5. Other tests cover:

- batch-row independence, including prompts of different lengths;
- `mx.compile` against eager execution;
- the batched guidance step against upstream's two separate passes;
- duplicate-prompt encoding, and the cached timestep tokens against per-step embedding;
- FP16/BF16 against FP32, and INT8;
- the converted-file round trip.

## Performance (Mac mini 2024, Apple M4 Pro: 20-core GPU, 64 GB)

| T = 120 frames (6 s) | N=1 latency | N=16 throughput |
|---|---|---|
| `kirin_text_image`, MLX FP32 / FP16 / BF16 | 317 / 232 / 244 ms | 8.5 / 10.0 / 9.6 motions/s |
| `kirin_text`, MLX FP32 / FP16 / BF16 | 275 / 285 / 285 ms | 8.5 / 10.0 / 10.2 motions/s |
| upstream PyTorch via cudashim (MPS), either model | 684–780 ms | 5.8–5.9 motions/s |

Details, INT8 and the analysis: [benchmarks/README.md](benchmarks/README.md).
