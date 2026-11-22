"""Fetch real DINOv3 image features and captions for a few AiM3D test clips, without the 14.5 GB zip.

Kirin's image-conditioned model takes a DINOv3 ViT-B/16 [CLS] feature. DINOv3 is gated on Hugging
Face, but upstream's AiM3D test set (``AiM_preview.zip``, the archive upstream's README downloads
for evaluation) ships the feature of every clip as ``mean_feature_dinov3.npy`` next to its
``captions.txt``. The server supports HTTP range requests, so this script reads only the zip's
central directory (once, ~15 MB; cached as JSON) and then the few members it needs, verifying
each member's CRC-32 and loading the arrays with ``allow_pickle=False``.

    uv run --all-packages python sandbox/scripts/fetch_aim_samples.py [--clips 8]

Writes ``sandbox/assets/aim/<species>/<clip>/{mean_feature_dinov3.npy,captions.txt}`` (gitignored;
AiM3D data is not redistributed here).
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
import zipfile
import zlib

import numpy as np
from cudashim.paths import SANDBOX

URL = "https://download.cs.stanford.edu/viscam/AiM/AiM_preview.zip"
OUT = SANDBOX / "assets" / "aim"
INDEX = SANDBOX / "assets" / "aim_preview_index.json"
WANTED = ("mean_feature_dinov3.npy", "captions.txt")
TIMEOUT_S = 120


def _get(start: int, end: int) -> bytes:
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "kirin-mlx"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        if response.status != 206:
            raise RuntimeError(f"server ignored the range request (HTTP {response.status})")
        return response.read()


class _RangeFile:
    """Minimal seekable, read-only file over HTTP ranges (enough for ``zipfile.ZipFile``)."""

    def __init__(self):
        request = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "kirin-mlx"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = _get(self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


def index() -> dict[str, dict]:
    """``{member name: zip fields}`` for the wanted files, from the cached JSON or the central directory."""
    if INDEX.is_file():
        return json.loads(INDEX.read_text())
    print("reading the zip's central directory over HTTP (~15 MB, once)...")
    with zipfile.ZipFile(_RangeFile()) as archive:
        entries = {info.filename: {"offset": info.header_offset, "compressed": info.compress_size,
                                   "size": info.file_size, "method": info.compress_type, "crc": info.CRC}
                   for info in archive.infolist() if info.filename.endswith(WANTED)}
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(entries))
    return entries


def extract(entry: dict) -> bytes:
    header = _get(entry["offset"], entry["offset"] + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("bad local file header")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = entry["offset"] + 30 + name_len + extra_len
    data = _get(start, start + entry["compressed"] - 1) if entry["compressed"] else b""
    if entry["method"] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif entry["method"] != zipfile.ZIP_STORED:
        raise RuntimeError(f"unsupported compression method {entry['method']}")
    if len(data) != entry["size"] or zlib.crc32(data) != entry["crc"]:
        raise RuntimeError("size or CRC-32 mismatch")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=int, default=8, help="number of clips, one per species first (default 8)")
    args = parser.parse_args()
    entries = index()
    clips = sorted({name.rsplit("/", 1)[0] for name in entries if name.endswith(WANTED[0])})
    by_species: dict[str, list[str]] = {}
    for clip in clips:
        by_species.setdefault(clip.split("/")[1], []).append(clip)
    chosen = []
    while len(chosen) < min(args.clips, len(clips)):  # round-robin over species
        for species in sorted(by_species):
            if by_species[species] and len(chosen) < args.clips:
                chosen.append(by_species[species].pop(0))
    for clip in chosen:
        target = OUT / clip.removeprefix("data/")
        target.mkdir(parents=True, exist_ok=True)
        for wanted in WANTED:
            name = f"{clip}/{wanted}"
            if name not in entries:
                continue
            data = extract(entries[name])
            (target / wanted).write_bytes(data)
        feature = np.load(target / WANTED[0], allow_pickle=False)
        if feature.shape != (1, 768) or not np.isfinite(feature).all():
            raise RuntimeError(f"{clip}: unexpected feature {feature.shape} {feature.dtype}")
        caption = (target / "captions.txt").read_text().splitlines()[0].split("#")[0].strip() \
            if (target / "captions.txt").is_file() else ""
        print(f"{clip.removeprefix('data/')}: feature {feature.shape}, caption {caption!r}")
    print(f"wrote {len(chosen)} clips to {OUT}")


if __name__ == "__main__":
    main()
