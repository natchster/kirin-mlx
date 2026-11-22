"""The checkpoint downloader: manifest, model selection and verification (offline, file:// URLs)."""

from __future__ import annotations

import hashlib

import pytest
from kirin_mlx import hub


def test_manifest_covers_both_models_and_text_encoder():
    entries = hub.manifest()
    paths = {e.path for e in entries}
    for model in ("kirin_text_image", "kirin_text"):
        for name in ("args.json", "model.pt", "Mean.npy", "Std.npy"):
            assert f"kirin/{model}/{name}" in paths
    for name in ("config.json", "model.safetensors", "vocab.txt"):
        assert f"distilbert-base-uncased/{name}" in paths
    assert all(len(e.sha256) == 64 and e.size > 0 and e.folder in hub.REPOS for e in entries)


def test_files_for_selects_model_and_shared_encoder():
    text = {e.path for e in hub.files_for("text")}
    assert any(p.startswith("kirin/kirin_text/") for p in text)
    assert not any(p.startswith("kirin/kirin_text_image/") for p in text)
    assert "distilbert-base-uncased/model.safetensors" in text
    with pytest.raises(ValueError, match="Unknown Kirin model"):
        hub.files_for("motionbricks")


def test_urls_are_pinned():
    entry = next(e for e in hub.manifest() if e.path.endswith("kirin_text/model.pt"))
    assert entry.url == f"https://huggingface.co/briannlongzhao/kirin/resolve/{hub.REPOS['kirin'][1]}/kirin_text/model.pt"


def _entry(tmp_path, content: bytes, sha=None, size=None):
    source = tmp_path / "source.bin"
    source.write_bytes(content)
    entry = hub.Entry(sha or hashlib.sha256(content).hexdigest(), len(content) if size is None else size,
                      "kirin/kirin_text/file.bin")
    return entry, source.as_uri()


def test_fetch_verifies_and_is_idempotent(tmp_path):
    entry, url = _entry(tmp_path, b"weights")
    dest = tmp_path / "dest"
    path = hub.fetch(entry, dest, quiet=True, url=url)
    assert path.read_bytes() == b"weights" and hub.verified(path, entry)
    assert hub.fetch(entry, dest, quiet=True, url="file:///nonexistent") == path  # verified: not downloaded again
    assert not list(dest.rglob("*.part"))


@pytest.mark.parametrize("kind", ["sha", "size"])
def test_fetch_rejects_corrupt_downloads(tmp_path, kind):
    entry, url = _entry(tmp_path, b"weights", sha="0" * 64 if kind == "sha" else None,
                        size=3 if kind == "size" else None)
    with pytest.raises(RuntimeError, match="checksum mismatch" if kind == "sha" else "size mismatch"):
        hub.fetch(entry, tmp_path / "dest", quiet=True, url=url)
    assert not (tmp_path / "dest" / entry.path).exists()
    assert not list((tmp_path / "dest").rglob("*.part"))


def test_fetch_reports_network_errors(tmp_path):
    entry, _ = _entry(tmp_path, b"weights")
    with pytest.raises(RuntimeError, match="rerun to retry"):
        hub.fetch(entry, tmp_path / "dest", quiet=True, url=(tmp_path / "missing.bin").as_uri())
