# SPDX-License-Identifier: MIT
"""Download and verify the released checkpoints, without the sandbox or extra dependencies.

Two Hugging Face repositories, each at the revision this port was tested with:

* ``briannlongzhao/kirin``: the two motion generators (``kirin_text``, ``kirin_text_image``),
  each a PyTorch ``model.pt`` plus ``args.json`` and the ``Mean.npy``/``Std.npy`` feature statistics;
* ``distilbert/distilbert-base-uncased``: the frozen text encoder Kirin conditions on (upstream
  loads it by name at run time; here it is pinned and verified like the rest).

Every file is checked against the size and SHA-256 in ``assets/checkpoints.sha256``. Files already
present and correct are not downloaded again. A download goes to a unique ``.part`` file next to
its target, is hashed while it streams, and is renamed into place only once its size and SHA-256
match, so an interrupted or corrupt download never leaves a file that looks complete.

    kirin-mlx download                         # both models + DistilBERT into checkpoints/
    kirin-mlx download --model kirin_text --dir /data/kirin
"""

from __future__ import annotations

import hashlib
import os
import secrets
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import canonical_name

# repository folder (first path component in the manifest) -> (Hugging Face repo, pinned revision)
REPOS = {
    "kirin": ("briannlongzhao/kirin", "c3a9114a1604e2fd5f1203df0dfc610137962a6c"),
    "distilbert-base-uncased": ("distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be"),
}
URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
MANIFEST = Path(__file__).resolve().parent / "assets" / "checkpoints.sha256"
UPSTREAM_COMMIT = "252a5ecae27383fb18757ac18da643cc809bbcd2"  # briannlongzhao/kirin (GitHub) this port follows
DEFAULT_DIR = Path("checkpoints")
TEXT_ENCODER = "distilbert-base-uncased"
TIMEOUT_S = 60  # a connection that delivers no data for this long fails instead of hanging
CHUNK = 1 << 20


@dataclass(frozen=True)
class Entry:
    sha256: str
    size: int
    path: str  # "<repository folder>/<path in the repository>", e.g. "kirin/kirin_text/model.pt"

    @property
    def folder(self) -> str:
        return self.path.split("/", 1)[0]

    @property
    def url(self) -> str:
        repo, revision = REPOS[self.folder]
        return URL.format(repo=repo, revision=revision, path=urllib.request.quote(self.path.split("/", 1)[1]))


def manifest(path: Path = MANIFEST) -> list[Entry]:
    entries = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            sha, size, relative = line.split()
            entries.append(Entry(sha, int(size), relative))
    return entries


def files_for(model: str) -> list[Entry]:
    """The files one model needs: its own folder plus the text encoder."""
    name = canonical_name(model)  # ValueError for an unknown model
    return [e for e in manifest() if e.path.startswith(f"kirin/{name}/") or e.folder == TEXT_ENCODER]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path: Path, entry: Entry) -> bool:
    return path.is_file() and path.stat().st_size == entry.size and sha256(path) == entry.sha256


def _stream(url: str, staged: Path) -> tuple[int, str]:
    """Download ``url`` into the new file ``staged``; returns (bytes, sha256 hex digest)."""
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={"User-Agent": "kirin-mlx"})
    with staged.open("xb") as out, urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def fetch(entry: Entry, dest: Path, quiet: bool = False, url: str | None = None) -> Path:
    """Download one file into ``dest / entry.path`` (via a unique ``.part`` file) and verify it.

    Raises ``RuntimeError`` naming the file on a network error, a size mismatch or a SHA-256
    mismatch; the partial file is removed, so rerunning retries cleanly."""
    target = Path(dest) / entry.path
    if verified(target, entry):
        if not quiet:
            print(f"{entry.path} (verified)")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Unique per attempt, so two concurrent downloads of the same file cannot interleave.
    staged = target.with_name(f"{target.name}.{os.getpid()}-{secrets.token_hex(4)}.part")
    if not quiet:
        print(f"{entry.path} ({entry.size / 1e6:.1f} MB)")
    url = url or entry.url
    try:
        size, digest = _stream(url, staged)
    except BaseException as exc:
        staged.unlink(missing_ok=True)
        if isinstance(exc, Exception):
            raise RuntimeError(f"downloading {entry.path} from {url} failed: {exc}; rerun to retry") from exc
        raise
    if size != entry.size:
        staged.unlink()
        raise RuntimeError(f"size mismatch for {entry.path}: got {size} bytes, expected {entry.size} "
                           "(truncated or wrong file); rerun to retry")
    if digest != entry.sha256:
        staged.unlink()
        raise RuntimeError(f"checksum mismatch for {entry.path}: sha256 {digest}, expected {entry.sha256}")
    staged.replace(target)
    return target


def download(models=("kirin_text_image", "kirin_text"), dest: str | Path = DEFAULT_DIR, quiet: bool = False) -> Path:
    """Fetch and verify the files of ``models`` (and the shared text encoder); returns ``dest``.

    ``dest / "kirin" / <model>`` is the folder :func:`kirin_mlx.load` takes, and
    ``dest / "distilbert-base-uncased"`` the text encoder it finds next to it."""
    dest = Path(dest)
    seen = set()
    for model in models:
        for entry in files_for(model):
            if entry.path not in seen:
                seen.add(entry.path)
                fetch(entry, dest, quiet)
    return dest


def checkpoint_dir(model: str, dest: str | Path = DEFAULT_DIR) -> Path:
    return Path(dest) / "kirin" / canonical_name(model)


def text_encoder_dir(dest: str | Path = DEFAULT_DIR) -> Path:
    return Path(dest) / TEXT_ENCODER
