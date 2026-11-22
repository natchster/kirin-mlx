# SPDX-License-Identifier: MIT
"""``kirin-mlx`` command line: download checkpoints, convert them, generate motion, quick benchmark."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import hub
from .config import FPS, MODELS, canonical_name
from .examples import PROMPTS
from .model import DTYPES, convert, load


def _model_or_all(name: str) -> str:
    return name if name == "all" else canonical_name(name)


def _load(args):
    if args.weights:
        model = load(args.weights, dtype=args.dtype)
    else:
        model = load(args.model, args.checkpoints or hub.checkpoint_dir(args.model), dtype=args.dtype)
    if getattr(args, "quantize", None):
        model.quantize(bits=args.quantize)
    return model


def _image_feature(path):
    if path is None:
        return None
    feature = np.load(path, allow_pickle=False)
    return feature.reshape(-1, feature.shape[-1]).astype(np.float32)


def cmd_download(args):
    models = MODELS if args.model == "all" else (args.model,)
    dest = hub.download(models, args.dir)
    for model in models:
        print(f"{model}: {hub.checkpoint_dir(model, dest)}")
    print(f"text encoder: {hub.text_encoder_dir(dest)}")


def cmd_convert(args):
    path = convert(args.model, args.checkpoints or hub.checkpoint_dir(args.model), args.output, dtype=args.dtype)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def cmd_generate(args):
    """Like upstream generate.py: results_<model>.npy (dict) + sample_NN.npy [T, 34, 3] per sample."""
    model = _load(args)
    image = _image_feature(args.image_feature)
    text = None if args.no_text else args.text_prompt
    start = time.perf_counter()
    out = model.generate(text, image_feature=image, num_samples=args.num_samples, motion_length=args.motion_length,
                         guidance=args.guidance_param, seed=args.seed)
    elapsed = time.perf_counter() - start
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"results_{model.spec.name}.npy",
            {"motions": out["joints"], "motion_vecs": out["features"], "texts": out["texts"], "lengths": out["lengths"],
             "num_samples": args.num_samples, "num_repetitions": 1})
    for i, (joints, length) in enumerate(zip(out["joints"], out["lengths"], strict=True)):
        np.save(out_dir / f"sample_{i:02d}.npy", joints[:int(length)])
    frames = int(out["lengths"][0])
    print(f"{model.spec.name} {args.dtype}: {args.num_samples} x {frames} frames ({frames / FPS:.1f} s) "
          f"in {elapsed * 1e3:.0f} ms -> {out_dir}/sample_NN.npy [T, 34, 3]")


def cmd_bench(args):
    model = _load(args)
    image = np.zeros((1, 768), np.float32) if model.spec.image_condition else None
    for batch in args.batch:
        def run(seed, batch=batch):
            model.generate(PROMPTS[0], image_feature=image, num_samples=batch, num_frames=args.frames, seed=seed)

        for i in range(args.warmup):
            run(i)
        times = []
        for i in range(args.iters):
            start = time.perf_counter()
            run(i)
            times.append(time.perf_counter() - start)
        ms = np.array(times) * 1e3
        print(f"{model.spec.name} {args.dtype} N={batch:3d} T={args.frames}: median {np.median(ms):8.1f} ms  "
              f"p90 {np.percentile(ms, 90):8.1f} ms  ({batch / np.median(ms) * 1e3:6.2f} motions/s, "
              f"{batch * args.frames / FPS / np.median(ms) * 1e3:6.1f}x real time)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kirin-mlx", description="Kirin animal motion generation on MLX")
    sub = parser.add_subparsers(dest="command", required=True)

    def model_args(p, need_source=True):
        p.add_argument("--model", type=canonical_name, choices=MODELS, default="kirin_text_image",
                       help="kirin_text_image (alias image) or kirin_text (alias text)")
        p.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
        p.add_argument("--checkpoints", help="model folder (default: checkpoints/kirin/<model>)")
        if need_source:
            p.add_argument("--weights", help="converted .safetensors from `kirin-mlx convert` "
                                             "(instead of --checkpoints)")
            p.add_argument("--quantize", type=int, choices=(4, 8), help="experimental: quantize the denoiser weights")

    p = sub.add_parser("download", help="download and verify the checkpoints (pinned revisions)")
    p.add_argument("--model", type=_model_or_all, choices=("all", *MODELS), default="all")
    p.add_argument("--dir", default=str(hub.DEFAULT_DIR), help="destination (default: checkpoints)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("convert", help="write one self-contained MLX .safetensors (denoiser + DistilBERT)")
    model_args(p, need_source=False)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("generate", help="generate motion from text and/or a DINOv3 image feature")
    model_args(p)
    p.add_argument("--text_prompt", "--text", default=PROMPTS[0])
    p.add_argument("--no_text", action="store_true", help="image-only generation (placeholder prompt ' ')")
    p.add_argument("--image_feature", "--image-feature", help="DINOv3 [CLS] feature .npy, shape [768] or [1, 768]")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--motion_length", type=float, default=6.0, help="seconds (max 9.8 = 196 frames)")
    p.add_argument("--guidance_param", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", default="out")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("bench", help="time complete generations")
    model_args(p)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
