"""Render a bench.py JSON as Markdown tables (latency in ms, motions/s in parentheses).

    uv run python model/benchmarks/report.py model/benchmarks/results/<file>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LABELS = {
    "mlx-fp32": "**MLX FP32** (GPU)",
    "mlx-fp16": "**MLX FP16** (GPU)",
    "mlx-bf16": "**MLX BF16** (GPU)",
    "mlx-int8": "MLX INT8 weights (GPU, experimental)",
    "torch-mps": "upstream PyTorch, `.cuda()`→MPS via cudashim",
    "torch-cpu": "upstream PyTorch, CPU",
}


def render(data: dict) -> str:
    host = data["host"]
    frames = data["results"][0]["frames"]
    lines = [f"Host: {host['chip']} ({host['gpu_cores']}-core GPU), {host['memory_gb']} GB, macOS {host['macos']}; "
             f"Python {host.get('python', '?')}, MLX {host['mlx']}, PyTorch {host.get('torch', '-')}. "
             f"T = {frames} frames ({frames / 20:.0f} s of motion at 20 fps); "
             f"image feature: {data['image_feature']}.", ""]
    for model in dict.fromkeys(r["model"] for r in data["results"]):
        rows = [r for r in data["results"] if r["model"] == model]
        batches = sorted({r["batch"] for r in rows})
        lines += [f"#### {model}: time per generation in ms (motions/s)", "",
                  "| backend | " + " | ".join(f"N={b}" for b in batches) + " |",
                  "|---|" + "---|" * len(batches)]
        for backend in dict.fromkeys(r["backend"] for r in rows):
            cells = []
            for b in batches:
                r = next((r for r in rows if r["backend"] == backend and r["batch"] == b), None)
                cells.append(f"{r['median_ms']:,.0f} ({r['motions_per_s']:.2f})" if r else "-")
            lines.append(f"| {LABELS.get(backend, backend)} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(json.loads(Path(sys.argv[1]).read_text())))
