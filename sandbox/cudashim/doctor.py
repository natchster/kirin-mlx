# SPDX-License-Identifier: MIT
"""``cudashim-doctor``: report which CUDA substitutes and sandbox pieces work on this Mac."""

from __future__ import annotations

import platform
import subprocess

from .paths import CHECKPOINTS, GOLDEN, MOTION_GENERATION, UPSTREAM


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def report() -> list[tuple[str, str, str]]:
    rows = []
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem = int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30
    rows.append(("host", "ok", f"{chip}, {mem:.0f} GB unified memory, macOS {platform.mac_ver()[0]}"))
    rows.append(("nvidia cuda", "absent", "no NVIDIA driver/GPU on Apple silicon; CUDA binaries cannot run"))

    try:
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        rows.append(("mlx (metal)", "ok", f"mlx {mx.__version__}, {info.get('device_name', '?')}"))
    except Exception as exc:
        rows.append(("mlx (metal)", "missing", str(exc)))

    try:
        import torch

        mps = torch.backends.mps.is_available()
        rows.append(("torch.cuda -> mps", "ok" if mps else "cpu-only", f"torch {torch.__version__}, mps={mps}"))
    except Exception as exc:
        rows.append(("torch.cuda -> mps", "missing", str(exc)))

    try:
        import transformers

        rows.append(("transformers", "ok", f"{transformers.__version__} (upstream pins 5.0.0)"))
    except Exception as exc:
        rows.append(("transformers", "missing", str(exc)))

    for module, why in (("xformers", "SpaTrackerV2/AniMer attention (CUDA-only wheels)"),
                        ("pytorch3d", "reconstruction + mesh fitting (CUDA build upstream)")):
        try:
            __import__(module)
            rows.append((module, "ok", why))
        except Exception:
            rows.append((module, "absent", f"not needed for motion generation; {why}"))

    commit = _run(["git", "-C", str(UPSTREAM), "rev-parse", "--short", "HEAD"])
    rows.append(("upstream reference", "ok" if (MOTION_GENERATION / "model/mdm.py").is_file() else "missing",
                 f"{UPSTREAM} @ {commit or '?'}"))
    try:
        from kirin_mlx import hub

        # Size check only (cheap); `kirin-mlx download --dir ...` verifies SHA-256 too.
        entries = hub.manifest()
        found = sum((CHECKPOINTS / e.path).is_file() and (CHECKPOINTS / e.path).stat().st_size == e.size
                    for e in entries)
        rows.append(("checkpoints", "ok" if found == len(entries) else "incomplete",
                     f"{found}/{len(entries)} files of the right size in {CHECKPOINTS}"))
    except Exception as exc:
        rows.append(("checkpoints", "unknown", str(exc)))
    fixtures = sorted(p.relative_to(GOLDEN).as_posix() for p in GOLDEN.glob("*/*.npz")) if GOLDEN.is_dir() else []
    rows.append(("golden fixtures", "ok" if fixtures else "missing", f"{len(fixtures)} in {GOLDEN}"))
    return rows


def main():
    rows = report()
    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"{name:<{width}}  {status:<11} {detail}")


if __name__ == "__main__":
    main()
