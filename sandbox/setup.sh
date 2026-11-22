#!/usr/bin/env bash
# One-shot bootstrap of the CUDA-emulation sandbox + reference oracle. Idempotent: verified
# downloads are skipped, and the fixtures are only produced when missing.
#
#   sandbox/setup.sh             # env, upstream @ pinned commit, checkpoints, golden fixtures
#   sandbox/setup.sh --aim       # also fetch real DINOv3 features + captions of 8 AiM3D test clips
#   sandbox/setup.sh --force     # regenerate every fixture
#
# Everything lands inside the repo (.venv/, sandbox/upstream/, sandbox/golden/, sandbox/assets/,
# checkpoints/) and every download is verified (SHA-256 for checkpoints, CRC-32 for AiM3D members).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

golden=1 force=0 aim=0
for arg in "$@"; do
  case "$arg" in
    --no-golden) golden=0 ;;
    --force) force=1 ;;
    --aim) aim=1 ;;
    *) echo "unknown option $arg (expected --no-golden, --force, --aim)" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "install uv: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]] || echo "warning: built for Apple silicon macOS" >&2

missing() {  # true when --force is given or any of the files is absent
  (( force )) && return 0
  local file
  for file in "$@"; do [[ -f "$file" ]] || return 0; done
  return 1
}

echo "==> uv environment (Python 3.12: MLX, PyTorch/MPS, transformers 5.0.0, upstream's import chain)"
uv sync --all-packages

echo "==> upstream source @ pinned commit, Hugging Face checkpoints (~500 MB, verified)"
uv run --all-packages python sandbox/scripts/fetch_assets.py

if (( golden )); then
  if missing sandbox/golden/text/{tokenizer,distilbert}.npz \
             sandbox/golden/kirin_text_image/{text_image,image_only,no_image,max_frames,per_sample}.npz \
             sandbox/golden/kirin_text/{text,mixed,unguided}.npz; then
    echo "==> golden fixtures from the upstream PyTorch generator (CPU, FP32)"
    uv run --all-packages python sandbox/scripts/make_golden.py
  else
    echo "==> golden fixtures present in sandbox/golden (--force regenerates)"
  fi
fi

if (( aim )); then
  echo "==> real DINOv3 features + captions of AiM3D test clips (range requests into AiM_preview.zip)"
  uv run --all-packages python sandbox/scripts/fetch_aim_samples.py
fi

echo "==> doctor"
uv run --all-packages cudashim-doctor
