# Benchmarks

`bench.py` times one complete generation per call, the way upstream `generate.py::generate` does
it for one batch:

1. DistilBERT text encoding;
2. 50 classifier-free-guided denoising steps (scale 2.5) from Gaussian noise;
3. de-normalization and joint recovery.

Every backend gets the same prompt, the same DINOv3 image feature (a real AiM3D one when
`sandbox/setup.sh --aim` has run), batch size N and motion length T = 120 frames (6 s, upstream's
default). The torch rows run Kirin's upstream PyTorch generator through the sandbox, with its own
device RNG. Results are in [`results/`](results/), and `report.py` renders them.

```sh
uv run --all-packages python model/benchmarks/bench.py --rounds 3                # all backends, N = 1 4 16
uv run --all-packages python model/benchmarks/bench.py --backends mlx-bf16 mlx-fp32 --batches 1 --frames 196
uv run python model/benchmarks/report.py model/benchmarks/results/<file>.json
```

`--rounds N` interleaves the backends: each runs `--iters` calls in turn, for N rounds with a
rotating start, and the statistics pool all rounds. Thermal drift then spreads evenly. Without the
sandbox, the MLX port alone can be timed with
`uv run --package kirin-mlx kirin-mlx bench --model kirin_text_image --dtype bfloat16 --batch 1 4 16`.

## Results

Host: Apple M4 Pro (20-core GPU), 64 GB, macOS 27.0; Python 3.12.11, MLX 0.32.4, PyTorch 2.15.0. T = 120 frames (6 s of motion at 20 fps); image feature: sandbox/assets/aim/bear/2AyBtNUWgg0_109_001/mean_feature_dinov3.npy.

#### kirin_text_image: time per generation in ms (motions/s)

| backend | N=1 | N=4 | N=16 |
|---|---|---|---|
| **MLX FP32** (GPU) | 317 (3.16) | 580 (6.89) | 1,891 (8.46) |
| **MLX FP16** (GPU) | 232 (4.32) | 553 (7.23) | 1,596 (10.02) |
| **MLX BF16** (GPU) | 244 (4.10) | 502 (7.97) | 1,668 (9.59) |
| MLX INT8 weights (GPU, experimental) | 308 (3.24) | 574 (6.97) | 2,508 (6.38) |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 780 (1.28) | 1,430 (2.80) | 2,701 (5.92) |
| upstream PyTorch, CPU | 1,415 (0.71) | 2,158 (1.85) | 5,815 (2.75) |

#### kirin_text: time per generation in ms (motions/s)

| backend | N=1 | N=4 | N=16 |
|---|---|---|---|
| **MLX FP32** (GPU) | 275 (3.64) | 647 (6.19) | 1,878 (8.52) |
| **MLX FP16** (GPU) | 285 (3.51) | 685 (5.84) | 1,596 (10.02) |
| **MLX BF16** (GPU) | 285 (3.51) | 508 (7.87) | 1,574 (10.17) |
| MLX INT8 weights (GPU, experimental) | 299 (3.35) | 632 (6.33) | 2,218 (7.21) |
| upstream PyTorch, `.cuda()`→MPS via cudashim | 684 (1.46) | 1,159 (3.45) | 2,738 (5.84) |
| upstream PyTorch, CPU | 1,427 (0.70) | 2,149 (1.86) | 5,738 (2.79) |

Each cell is the median of 12 timed calls (3 interleaved rounds × 4; 6 for PyTorch on the CPU)
after warm-up. Full statistics, including p10/p90 and first-call times, are in the JSON. Upstream
publishes no inference timings, so there is no CUDA reference row.

## Reading the numbers

- **Generation is compute-bound, even at N = 1.** One sample is 120 frame tokens, and guidance
  doubles that to 240 per network pass. At about 45 MFLOP per token per pass, a single generation
  is about **540 GFLOP** (50 steps). MLX does that in 0.23–0.32 s, which is 1.7–2.3 TFLOPS. The GEMM peak
  of this GPU is 6.2 TFLOPS in FP32 and 7.2 TFLOPS in FP16/BF16 (measured). The gap at N = 1 is the
  dependent chain of about 130 small kernels per pass. At N = 16 the kernels are large enough:
  BF16 reaches **5.2–5.5 TFLOPS** effective, close to peak.
- **So smaller weights barely help.** The weights are only 107 MB in FP32. BF16 is 12–21% faster
  than FP32 at N ≥ 4 through faster GEMMs. FP16 is usually similar but not consistently faster, and
  at N = 1 the precisions are within run-to-run noise of each other. INT8 weights gain nothing at N = 1 and are slower at N = 16, where MLX's
  quantized GEMM is compute-bound. This is the opposite of MotionBricks, whose ~800 MB of weights
  made batch 1 bandwidth-bound.
- **Batching is the throughput lever.** N = 16 costs 5.5–7× the time of N = 1 for 16× the motions:
  about 10 six-second motions per second in BF16, **60× faster than real time**.
- **MLX vs the upstream code on the same Mac:**
  - against PyTorch on the GPU (MPS, through cudashim): 2.4–3.4× faster at N = 1, 1.6–1.7× at N = 16 (BF16);
  - against PyTorch on the CPU (14 cores): 4.5–6× faster at N = 1, 3.5–3.6× at N = 16.
  
  The margin over MPS shrinks with batch size because both become GEMM-bound. At N = 1, MLX's
  compiled steps, single-batch guidance (upstream makes two network calls per step) and lazily
  queued 50-step graph remove most of PyTorch's per-op overhead.
- **Tails are long at N = 1.** p90 is up to 2× the median, depending on whether the GPU clocks
  had dropped between calls; interleaving with the CPU-heavy torch-cpu backend makes that worse.
  Use `--rounds` and compare medians. The table is therefore pessimistic for MLX at N = 1: timed
  alone (`kirin-mlx bench --model kirin_text --dtype bfloat16 --batch 1`), a generation takes
  **166 ms** (p90 170 ms), 36× faster than real time.
- **First calls** include `mx.compile` tracing (MLX) or MPS kernel compilation (PyTorch), and cost
  about one extra generation.

## Optimizations applied, with measured effect

| Change | Effect |
|---|---|
| Classifier-free guidance as one `2N` batch instead of two network calls | fewer, larger kernels; equal to the two-pass result to ≤ 1e-5 (`test_batched_guidance_equals_two_passes`) |
| Timestep tokens of all 50 steps computed once per model | removes 2 GEMMs + SiLU per step |
| One `mx.compile`-d graph per step; 50 steps queued lazily, one `mx.eval` | 50-step loop 267 ms vs 357 ms with an `mx.eval` per step (FP32, N = 1) |
| DistilBERT once per call, on unique prompts only | text encoding 2.5 ms, independent of N for a shared prompt |
| BF16 denoiser (`dtype="bfloat16"`) | 12–21% at N ≥ 4 (table above); FP16 similar but less consistent |
| INT8 weights (`quantize(8)`) | no gain: compute-bound; kept as an option |

What would help next: N = 1 runs at a third of GEMM peak, so a model-specific runtime that fuses
the per-layer kernel chain should bring single-motion latency down further. That is the role the
Rust Metal runtime plays in [motionbricks-mlx](https://github.com/natchster/motionbricks-mlx), where it gave 2× at batch 1.
