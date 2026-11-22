"""Latency / throughput of Kirin motion generation on Apple silicon across backends.

One timed call = one complete generation, as upstream ``generate.py::generate`` does it for one
batch: DistilBERT text encoding, 50 classifier-free-guided denoising steps (scale 2.5) from
Gaussian noise, de-normalization and joint recovery. Every backend gets the same prompt, the same
(real or synthetic) DINOv3 image feature, the same batch size N and motion length T.

Backends:

* ``mlx-fp32`` / ``mlx-fp16`` / ``mlx-bf16``  the MLX port, compiled steps, Metal GPU
* ``mlx-int8``                              MLX, 8-bit denoiser weights (experimental)
* ``torch-mps``                             upstream PyTorch model via cudashim (``.cuda()`` -> MPS)
* ``torch-cpu``                             upstream PyTorch model on the CPU, FP32

    uv run --all-packages python model/benchmarks/bench.py --models kirin_text_image --batches 1 4 16

``--rounds N`` interleaves the backends (each runs ``--iters`` calls in turn, N rounds with a
rotating start, statistics pooled), so thermal drift spreads evenly over them.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sandbox"))

import kirin_mlx as kirin  # noqa: E402
import mlx.core as mx  # noqa: E402
from kirin_mlx.examples import PROMPTS, image_feature  # noqa: E402

CKPT = ROOT / "checkpoints"
AIM = ROOT / "sandbox" / "assets" / "aim"
PROMPT = PROMPTS[0]


def feature_for_bench() -> tuple[np.ndarray, str]:
    """A real AiM3D DINOv3 feature when ``fetch_aim_samples.py`` has run, else a synthetic one."""
    found = sorted(AIM.glob("*/*/mean_feature_dinov3.npy"))
    if found:
        return np.load(found[0], allow_pickle=False).astype(np.float32), str(found[0].relative_to(ROOT))
    return image_feature(0), "synthetic (seed 0)"


def sample(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.array(times) * 1e3


def stats(ms: np.ndarray) -> dict:
    return {"median_ms": float(np.median(ms)), "p10_ms": float(np.percentile(ms, 10)),
            "p90_ms": float(np.percentile(ms, 90))}


def mlx_runner(name, dtype, bits=None):
    model = kirin.load(name, CKPT / "kirin" / name, dtype=dtype)
    if bits:
        model.quantize(bits=bits)

    def make(batch, frames, feature):
        image = feature if model.spec.image_condition else None
        counter = iter(range(1 << 30))

        def run():
            model.generate(PROMPT, image_feature=image, num_samples=batch, num_frames=frames, seed=next(counter))

        return run

    return make


def torch_runner(name, device):
    """Upstream ``generate.py::generate`` for one batch, with upstream's own (device) RNG."""
    import torch
    from cudashim import reference, torch_cuda

    if device == "mps":
        torch_cuda.enable()
        dev = "cuda"
    else:
        dev = "cpu"
    ref = reference.build(name, dev, image=kirin.get_spec(name).image_condition)
    reference.import_upstream()
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from data_loaders.tensors import collate

    mean = np.load(reference.checkpoint_dir(name) / "Mean.npy")
    std = np.load(reference.checkpoint_dir(name) / "Std.npy")

    def make(batch, frames, feature):
        def run():
            model = ref.model
            shape = (batch, ref.mdm.njoints, ref.mdm.nfeats, frames)
            item = {"inp": torch.zeros(frames), "tokens": None, "lengths": frames, "text": PROMPT}
            _, kwargs = collate([item] * batch)
            kwargs["y"] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in kwargs["y"].items()}
            kwargs["y"]["scale"] = torch.ones(batch, device=dev) * ref.args.guidance_param
            kwargs["y"]["text_embed"] = model.encode_text(kwargs["y"]["text"])
            if ref.image_condition:
                kwargs["y"]["feature"] = torch.as_tensor(feature.reshape(1, 768), device=dev).repeat(batch, 1, 1)
            with torch.no_grad():
                out = ref.diffusion.p_sample_loop(model, shape, clip_denoised=False, model_kwargs=kwargs,
                                                  skip_timesteps=0, init_image=None, progress=False, dump_steps=None,
                                                  noise=None, const_noise=False)
                features = (out.cpu().permute(0, 2, 3, 1) * std + mean).float()
                recover_from_ric(features, 34)

        return run

    return make


BACKENDS = {
    "mlx-fp32": lambda n: mlx_runner(n, "float32"),
    "mlx-fp16": lambda n: mlx_runner(n, "float16"),
    "mlx-bf16": lambda n: mlx_runner(n, "bfloat16"),
    "mlx-int8": lambda n: mlx_runner(n, "float32", bits=8),
    "torch-mps": lambda n: torch_runner(n, "mps"),
    "torch-cpu": lambda n: torch_runner(n, "cpu"),
}
SLOW = {"torch-cpu"}


def host():
    sysctl = lambda key: subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()  # noqa: E731
    gpu_cores = subprocess.run(
        "system_profiler SPDisplaysDataType | awk -F': ' '/Total Number of Cores/{print $2; exit}'",
        shell=True, capture_output=True, text=True).stdout.strip()
    info = {"chip": sysctl("machdep.cpu.brand_string"), "memory_gb": int(sysctl("hw.memsize")) // 2**30,
            "gpu_cores": gpu_cores, "macos": platform.mac_ver()[0], "python": platform.python_version(),
            "mlx": mx.__version__}
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["kirin_text_image", "kirin_text"])
    parser.add_argument("--backends", nargs="+", default=["mlx-fp32", "mlx-fp16", "mlx-bf16", "torch-mps", "torch-cpu"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--frames", type=int, default=120, help="motion length in frames at 20 fps (default 120 = 6 s)")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1, help="interleave the backends over this many rounds")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    info = host()
    feature, feature_source = feature_for_bench()
    print(f"# {info['chip']} ({info['gpu_cores']}-core GPU), {info['memory_gb']} GB, macOS {info['macos']}, "
          f"MLX {info['mlx']}, torch {info.get('torch', '-')}; T={args.frames} frames; image feature: {feature_source}")
    results = []
    for name in args.models:
        makers = {}
        for backend in args.backends:
            try:
                start = time.perf_counter()
                makers[backend] = (BACKENDS[backend](name), time.perf_counter() - start)
            except Exception as exc:
                print(f"{name:16s} {backend:10s} unavailable: {exc}")
        for batch in args.batches:
            runs, first_ms = {}, {}
            for backend, (make, _) in makers.items():
                runs[backend] = make(batch, args.frames, feature)
                start = time.perf_counter()
                runs[backend]()  # includes graph compilation / MPS kernel warm-up
                first_ms[backend] = (time.perf_counter() - start) * 1e3
            order = list(runs)
            times = {backend: [] for backend in order}
            for r in range(args.rounds):
                for backend in order[r % len(order):] + order[:r % len(order)]:
                    iters = max(2, args.iters // 4) if backend in SLOW else args.iters
                    times[backend].append(sample(runs[backend], 1 if backend in SLOW else args.warmup, iters))
            for backend in order:
                result = stats(np.concatenate(times[backend]))
                motion_s = batch * args.frames / kirin.FPS
                result.update(model=name, backend=backend, batch=batch, frames=args.frames, rounds=args.rounds,
                              first_call_ms=first_ms[backend], setup_s=makers[backend][1],
                              motions_per_s=batch / result["median_ms"] * 1e3,
                              realtime_factor=motion_s / (result["median_ms"] / 1e3))
                results.append(result)
                print(f"{name:16s} {backend:10s} N={batch:3d}  median {result['median_ms']:9.1f} ms  "
                      f"p90 {result['p90_ms']:9.1f} ms  {result['motions_per_s']:7.2f} motions/s  "
                      f"{result['realtime_factor']:7.1f}x real time  (first call {first_ms[backend]:7.0f} ms)")
        if "torch-mps" in makers:
            from cudashim import torch_cuda

            torch_cuda.disable()
    output = args.output or ROOT / "model/benchmarks/results" / f"{date.today()}-{info['chip'].replace(' ', '-')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"host": info, "image_feature": feature_source, "prompt": PROMPT,
                                  "results": results}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
