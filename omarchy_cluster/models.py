"""Local Hugging Face model inventory and download.

Harbor-parity gap (issue 45): the cluster could only read an already
populated HF cache. This module lists what is loadable and pulls a repo
into the same cache layout planner.find_model_path already resolves, so
`models pull` feeds straight into `serve --model <repo>`.
"""
from __future__ import annotations

import os


def cache_dir() -> str:
    return os.path.expanduser("~/.cache/huggingface/hub")


def _repo_from_dir(name):
    if not name.startswith("models--"):
        return None
    return name[len("models--"):].replace("--", "/")


def _snapshot_info(snap):
    """Largest-file summary of one snapshot: what serve would actually load."""
    total = 0
    weight_files = 0
    for root, _, files in os.walk(snap):
        for f in files:
            p = os.path.join(root, f)
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
            if f.endswith((".safetensors", ".gguf", ".npz", ".bin")):
                weight_files += 1
    return total, weight_files


def list_local():
    """Every cached repo with a config.json snapshot: [{repo, path, bytes, weights}]."""
    out = []
    base = cache_dir()
    try:
        entries = sorted(os.listdir(base))
    except (FileNotFoundError, NotADirectoryError):
        return out
    for name in entries:
        repo = _repo_from_dir(name)
        if repo is None:
            continue
        snaps = os.path.join(base, name, "snapshots")
        try:
            revisions = sorted(os.listdir(snaps))
        except (FileNotFoundError, NotADirectoryError):
            continue
        for entry in revisions:
            snap = os.path.join(snaps, entry)
            if not os.path.isfile(os.path.join(snap, "config.json")):
                continue
            total, weights = _snapshot_info(snap)
            out.append({"repo": repo, "path": snap, "bytes": total, "weights": weights})
            break
    return out


def _downloader():
    import huggingface_hub
    return huggingface_hub.snapshot_download


def pull(repo, progress=None):
    """Download a HF repo into the cache; returns the snapshot path.

    huggingface_hub is imported lazily: nodes that only serve (never
    download) do not pay the dependency, and the error names the fix.
    """
    try:
        download = _downloader()
    except ImportError as e:
        raise SystemExit("models pull needs huggingface_hub (pip install huggingface_hub): {}".format(e))
    path = download(repo_id=repo)
    if progress is not None:
        total, weights = _snapshot_info(path)
        progress({"repo": repo, "path": path, "bytes": total, "weights": weights})
    return path
