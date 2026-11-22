"""Generate golden fixtures with the upstream Kirin PyTorch generator (the reference oracle).

``sandbox/golden/text/``: DistilBERT tokenization of tricky strings and hidden states of a padded
batch, from the Hugging Face tokenizer/model exactly as upstream loads them.

``sandbox/golden/<model>/<case>.npz``: one generation per case through upstream's sampler, with the
noise drawn from ``NoiseSource(seed)`` (so the MLX side can rebuild it), storing the inputs, the
text encoding, the denoiser's per-layer tensors at t = 49, the sampler state and network outputs
at several timesteps, and the final normalized sample, de-normalized features and joints.
``meta.json`` records the device and versions that produced them.

Only CPU fixtures are the FP32 reference the tests expect, so ``--device mps`` writes to
``sandbox/golden-mps/`` unless ``--out`` says otherwise.

    uv run --all-packages python sandbox/scripts/make_golden.py [--device cpu|mps] [--out DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from cudashim import reference
from cudashim.paths import GOLDEN
from kirin_mlx.examples import PROMPTS, image_feature

TOKENIZER_STRINGS = (
    "a horse rears up on its hind legs",
    "The CAT walks slowly, turns around — and sits down!",
    "Café naïve résumé façade Ünïcödé ÅNGSTRÖM",
    "日本の犬が走る and 中文字",
    "emoji \U0001f40e and symbols ©®™ ± ½",
    "  multiple   spaces\tand\nnewlines  nbsp　ideographic  ",
    "unaffable antidisestablishmentarianism quadrupedal galloping",
    "a [MASK] dog [SEP] runs [cls] [UNK]",
    "control\x00chars\x07here​zero‍width�",
    "x" * 120 + " short",
    " ",
    "",
    "It's 3.5km/h, isn't it? (yes) [no] {maybe} #tag @user 100% $5 ~ok~",
    "İstanbul Σίσυφος ΟΔΥΣΣΕΥΣ ﬁ ligature",
    "the giraffe bends its long neck down to drink water, then lifts its head and walks away slowly",
)

CASES = {
    "kirin_text_image": {
        "text_image": dict(texts=[PROMPTS[0]] * 2, image=image_feature(10), frames=120, seed=1),
        "image_only": dict(texts=[" "] * 2, image=image_feature(11), frames=60, seed=2),  # upstream --no_text
        "no_image": dict(texts=[PROMPTS[1], PROMPTS[2]], image=None, frames=60, seed=3),  # upstream --no_image
        "max_frames": dict(texts=[PROMPTS[3]], image=image_feature(12), frames=196, seed=4),
        "per_sample": dict(texts=[PROMPTS[0], PROMPTS[2]], image=image_feature(13, batch=2), frames=40, seed=5),
    },
    "kirin_text": {
        "text": dict(texts=[PROMPTS[1]] * 2, image=None, frames=120, seed=6),
        "mixed": dict(texts=[PROMPTS[0], PROMPTS[2], PROMPTS[3]], image=None, frames=60, seed=7),
        "unguided": dict(texts=[PROMPTS[1], PROMPTS[3]], image=None, frames=60, seed=8, guidance=1.0),
    },
}


def text_fixtures(out_dir: Path) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(reference.text_encoder_dir()))
    ids = [tokenizer(s)["input_ids"] for s in TOKENIZER_STRINGS]
    np.savez_compressed(out_dir / "tokenizer.npz", texts=np.array(TOKENIZER_STRINGS),
                        ids=np.concatenate(ids).astype(np.int32),
                        offsets=np.cumsum([0] + [len(i) for i in ids]).astype(np.int64))
    model = AutoModel.from_pretrained(str(reference.text_encoder_dir())).eval()
    texts = [*PROMPTS, " ", TOKENIZER_STRINGS[-1]]
    batch = tokenizer(texts, return_tensors="pt", padding=True)  # upstream BERT.forward
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    np.savez_compressed(out_dir / "distilbert.npz", texts=np.array(texts),
                        input_ids=batch.input_ids.numpy().astype(np.int32),
                        keep=batch.attention_mask.numpy().astype(bool), hidden=hidden.numpy())
    reference.write_meta(out_dir, reference.fixture_meta("sandbox/scripts/make_golden.py", "cpu"))
    print(f"text: {len(TOKENIZER_STRINGS)} tokenizer strings, DistilBERT batch {tuple(hidden.shape)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--models", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--out", type=Path, help="output root (default: sandbox/golden, or sandbox/golden-<device>)")
    args = parser.parse_args()
    out_root = args.out or (GOLDEN if args.device == "cpu" else GOLDEN.with_name(f"golden-{args.device}"))
    if args.device == "cpu":
        text_fixtures(out_root / "text")
    device = args.device
    if device == "mps":
        from cudashim import torch_cuda

        torch_cuda.enable()
        device = "cuda"  # NVIDIA-style code path, lands on MPS
    for name in args.models:
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        refs = {}
        for case, spec in CASES[name].items():
            image, guidance = spec["image"] is not None, spec.get("guidance", 2.5)
            if (image, guidance) not in refs:
                start = time.perf_counter()
                refs[image, guidance] = reference.build(name, device, image=image, guidance=guidance)
                print(f"{name}: built upstream reference (image={image}, guidance={guidance}) "
                      f"in {time.perf_counter() - start:.1f}s")
            ref = refs[image, guidance]
            start = time.perf_counter()
            result = reference.run(ref, spec["texts"], spec["frames"], spec["seed"], spec["image"], intermediates=True)
            inputs = {"in.texts": np.array(spec["texts"]), "in.num_frames": np.int64(spec["frames"]),
                      "in.seed": np.int64(spec["seed"]), "in.guidance": np.float64(guidance)}
            if image:
                inputs["in.image_feature"] = spec["image"]
            np.savez_compressed(out_dir / f"{case}.npz", **inputs, **result)
            print(f"  {case}: B={len(spec['texts'])} T={spec['frames']} in {time.perf_counter() - start:.1f}s, "
                  f"joints range [{result['joints'].min():.2f}, {result['joints'].max():.2f}]")
        reference.write_meta(out_dir, reference.fixture_meta("sandbox/scripts/make_golden.py", args.device,
                                                             cases=list(CASES[name])))


if __name__ == "__main__":
    main()
