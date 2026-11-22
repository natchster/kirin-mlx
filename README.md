# kirin-mlx

The motion generator of [Kirin](https://github.com/briannlongzhao/kirin) (*Animal Motion Generation
from In-the-Wild Video*, ECCV 2026) on Apple silicon. Kirin generates 3D motion for quadruped
animals from a text prompt, an image of the animal, or both: up to 9.8 s at 20 fps, as 34-joint
skeleton positions. It is a 50-step diffusion model (an MDM-style transformer decoder) conditioned
on DistilBERT text features and a DINOv3 image feature, trained on the AiM3D dataset.

This repository runs it on the Apple GPU with **MLX**, and runs Kirin's original CUDA PyTorch code
on the same Mac as the reference.

> [!IMPORTANT]
> **For now, only Kirin's motion generator is ported.** That is the model that turns text and/or an
> image of an animal into 3D motion (upstream `motion_generation/`, the part with released weights).
> The rest of Kirin is **not** ported yet:
>
> - **video reconstruction** (`run_reconstruction.py`: AniMer, SpaTrackerV2, SAM 3), which recovers
>   3D motion from videos;
> - **mesh animation** (`animate_mesh.py`: a Rodin-generated mesh skinned to the motion);
> - the **DINOv3 image encoder**. Image conditioning takes a precomputed 768-d DINOv3 feature, as
>   upstream's `--image_feature_path` does, so you can't pass an image file directly yet.
>
> [sandbox/README.md](sandbox/README.md#the-short-answer-from-the-research) explains why, stage by
> stage.

> [!NOTE]
> **This is a quick port and may contain issues.** It passes a test suite that checks it stage by
> stage against Kirin's own PyTorch code. If you find a discrepancy with upstream, please open an
> issue. This is an independent community project, not affiliated with or endorsed by the Kirin
> authors.

| Folder | What it is |
|---|---|
| [`model/`](model/README.md) | **`kirin_mlx`**, the MLX port of the **motion generator**, both released models (`kirin_text_image`, `kirin_text`): DistilBERT, the denoiser, the guided sampler and joint recovery, in FP32, FP16, BF16 or INT8. It depends only on `mlx` and `numpy`; there's no PyTorch even to read the `.pt` checkpoints. |
| [`sandbox/`](sandbox/README.md) | **`cudashim`**, a CUDA-emulation layer that runs Kirin's original PyTorch code, including its unmodified `generate.py`, on a Mac. It is the reference the port is tested against, and it is only needed for development. Its README also surveys the stages that are not ported (reconstruction, mesh animation). |

Ported from upstream commit `252a5ec` (motion generator only), with the weights at Hugging Face
revision `c3a9114`. The approach follows [motionbricks-mlx](https://github.com/natchster/motionbricks-mlx).

## Performance

Median time per generation (one call = text encoding + 50 guided denoising steps + joint
recovery, T = 120 frames = 6 s of motion) on a Mac mini with an Apple M4 Pro (20-core GPU), with
motions/s in parentheses:

| Backend | `kirin_text_image` N=1 | N=16 | `kirin_text` N=1 | N=16 |
|---|---|---|---|---|
| **Python MLX, BF16** | 244 ms (4.1) | 1.67 s (9.6) | 285 ms (3.5) | 1.57 s (10.2) |
| **Python MLX, FP16** | 232 ms (4.3) | 1.60 s (10.0) | 285 ms (3.5) | 1.60 s (10.0) |
| Python MLX, FP32 | 317 ms (3.2) | 1.89 s (8.5) | 275 ms (3.6) | 1.88 s (8.5) |
| upstream PyTorch on the GPU (MPS, via cudashim) | 780 ms (1.3) | 2.70 s (5.9) | 684 ms (1.5) | 2.74 s (5.8) |
| upstream PyTorch on the CPU | 1.42 s (0.7) | 5.81 s (2.8) | 1.43 s (0.7) | 5.74 s (2.8) |

- **MLX vs upstream:** 2.4–3.4× faster than Kirin's PyTorch model on the same GPU at N = 1, and
  1.6–1.7× at N = 16.
- **Real time:** one 6-second motion takes about a quarter of a second. Batches of 16 sustain about
  10 motions/s, **60× faster than real time**.
- **Compute-bound:** each sample is 240 frame tokens per guided pass, about 540 GFLOP per
  generation. Batching therefore matters more than precision, and INT8 weights don't help.

[All numbers, INT8, N = 4 and the analysis](model/benchmarks/README.md) are in the benchmark notes.

## Use the model

The model takes a prompt and/or a DINOv3 image feature and returns joint positions. Downloading
the checkpoints (both models plus DistilBERT, about 500 MB) verifies every file's SHA-256. No
sandbox is needed.

```sh
git clone https://github.com/natchster/kirin-mlx && cd kirin-mlx
uv sync --package kirin-mlx                       # Python 3.12 + uv; or: pip install ./model
uv run --package kirin-mlx kirin-mlx download
```

```python
import kirin_mlx as kirin

model = kirin.load("kirin_text_image", "checkpoints/kirin/kirin_text_image", dtype="bfloat16")
out = model.generate("a horse rears up on its hind legs", image_feature=feature, num_samples=3, seed=0)
out["joints"]          # [3, 120, 34, 3]: world-space joint positions, 6 s at 20 fps
```

`feature` is a DINOv3 ViT-B/16 [CLS] feature of shape `[768]`. DINOv3 is gated on Hugging Face,
so `sandbox/setup.sh --aim` fetches real features of AiM3D test clips. `image_feature=None` gives
text-only generation, and `text=None` gives image-only.

```sh
uv run --package kirin-mlx kirin-mlx generate --model kirin_text --text "a dog runs forward and then jumps" --num_samples 3
uv run --package kirin-mlx kirin-mlx generate --model kirin_text_image --image-feature mean_feature_dinov3.npy \
    --text "a brown bear walks slowly on the beach" --dtype bfloat16 --output_dir out/bear
```

Outputs use upstream's format: `sample_NN.npy` (`[T, 34, 3]`) and `results_<model>.npy`. Upstream's
`render_motion.py` and `view_motion.py` can display them. See the [model README](model/README.md)
for the API, the converted single-file weights and the design.

## Benchmark and check correctness

```sh
sandbox/setup.sh --aim                                    # upstream @ 252a5ec, golden fixtures, AiM3D features
uv run --all-packages pytest                              # 123 tests (model + sandbox)
uv run --all-packages python model/benchmarks/bench.py --rounds 3   # MLX vs upstream PyTorch (MPS, CPU)
uv run --package kirin-mlx kirin-mlx bench --model kirin_text_image --dtype bfloat16   # MLX only
```

The model suite also runs without the sandbox; only the parity tests against upstream skip. Every
stage is checked against Kirin's PyTorch code with the same inputs and noise:

- the tokenizer gives identical ids;
- DistilBERT and each decoder layer agree to within 2e-6;
- one denoising step agrees to within 5e-6;
- a full 50-step generation agrees to within 7.4e-4.

Details are in [model/README.md](model/README.md#verification).

## License

The code is **MIT** ([LICENSE](LICENSE)), like upstream's. [NOTICE](NOTICE) credits Kirin, MDM,
guided-diffusion, HumanML3D and Hugging Face transformers, whose code the port re-implements.

Not included in this repository:

- **Weights:** downloaded from [briannlongzhao/kirin](https://huggingface.co/briannlongzhao/kirin)
  (MIT per the model card) and
  [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) (Apache-2.0).
- **AiM3D data:** the optional test-clip features are fetched for local use only.

## Citation

```bibtex
@inproceedings{zhao2026kirin,
  title={Kirin: Animal Motion Generation from In-the-Wild Video},
  author={Zhao, Brian Nlong and Pan, Zhuoyang and Rehg, James M. and Wu, Jiajun and Wu, Shangzhe},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
