"""Fetch everything the sandbox needs, verifying every byte against a pinned digest.

* Upstream source: briannlongzhao/kirin cloned at a pinned commit into sandbox/upstream/.
  ``--submodules`` also checks out AniMer and SpaTrackerV2 (video reconstruction only; the
  motion generator does not need them).
* Checkpoints: both generators and DistilBERT from Hugging Face at pinned revisions, fetched by
  the model package's own downloader (``kirin_mlx.hub``, also usable without the sandbox), which
  checks each file's size and SHA-256 against ``model/kirin_mlx/assets/checkpoints.sha256``.

Usage (from the repo root):  uv run --all-packages python sandbox/scripts/fetch_assets.py [--submodules]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from cudashim.paths import CHECKPOINTS, UPSTREAM
from kirin_mlx import hub

UPSTREAM_REPO = "https://github.com/briannlongzhao/kirin.git"
UPSTREAM_COMMIT = hub.UPSTREAM_COMMIT


def git(*args, **kwargs):
    return subprocess.run(["git", *args], check=True, **kwargs)


def clone_upstream(submodules: bool) -> None:
    if (UPSTREAM / ".git").exists():
        head = git("-C", str(UPSTREAM), "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
        if head != UPSTREAM_COMMIT:
            raise SystemExit(f"{UPSTREAM} is at {head}, expected {UPSTREAM_COMMIT}. Remove it and rerun.")
        print(f"upstream: already at {UPSTREAM_COMMIT[:12]}")
    else:
        UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--quiet", "--filter=blob:none", UPSTREAM_REPO, str(UPSTREAM))
        git("-C", str(UPSTREAM), "checkout", "--quiet", UPSTREAM_COMMIT)
        print(f"upstream: cloned at {UPSTREAM_COMMIT[:12]}")
    if submodules:
        git("-C", str(UPSTREAM), "submodule", "update", "--init", "--quiet", "--filter=blob:none")
        print("upstream: submodules AniMer, SpaTrackerV2 checked out at the pinned commits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--submodules", action="store_true", help="also check out AniMer and SpaTrackerV2")
    parser.add_argument("--no-checkpoints", action="store_true", help="skip the Hugging Face checkpoints")
    args = parser.parse_args()
    try:
        clone_upstream(args.submodules)
        if not args.no_checkpoints:
            hub.download(dest=CHECKPOINTS)
    except RuntimeError as exc:  # download / verification failures from kirin_mlx.hub
        raise SystemExit(f"error: {exc}") from exc
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
