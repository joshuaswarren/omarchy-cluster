"""Local Hugging Face model inventory and download.

Harbor-parity gap (issue 45): the cluster could only read an already
populated HF cache. This module lists what is loadable and pulls a repo
into the same cache layout planner.find_model_path already resolves, so
`models pull` feeds straight into `serve --model <repo>`.
"""
from __future__ import annotations

import json
import os
import re

# Hugging Face's rule for a repo id: "org/name", each part starting and ending with a word character,
# dots and single dashes inside, at most 96 characters, no ".." and no ".git" ending.
_PART = r"\w(?:[\w.-]*\w)?"
_REPO_ID = re.compile(_PART + "/" + _PART)


def valid_repo(repo):
    return (isinstance(repo, str) and len(repo) <= 96 and _REPO_ID.fullmatch(repo) is not None
            and "--" not in repo and ".." not in repo and not repo.endswith(".git"))


class ModelsError(RuntimeError):
    """A models operation failed for a reason the caller can say in words (library code never exits the process)."""


def cache_dir() -> str:
    """The Hugging Face hub cache, where huggingface_hub itself puts it: HF_HUB_CACHE, else $HF_HOME/hub, else the default."""
    explicit = os.environ.get("HF_HUB_CACHE")
    home = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    return os.path.expanduser(explicit or os.path.join(home, "hub"))


def pick_snapshot(repo_dir):
    """The snapshot serve should load for one cached repo directory: what refs/main names (and nothing else when
    that ref exists, so an updated repo never falls back to old weights), else the newest complete snapshot by
    modification time. None when there is none."""
    snaps = os.path.join(repo_dir, "snapshots")
    main_ref = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(main_ref):
        with open(main_ref) as f:
            snap = os.path.join(snaps, f.read().strip())
        return snap if snapshot_complete(snap) else None
    try:
        names = os.listdir(snaps)
    except (FileNotFoundError, NotADirectoryError):
        return None
    complete = [os.path.join(snaps, n) for n in names if snapshot_complete(os.path.join(snaps, n))]
    return max(complete, key=lambda p: (os.path.getmtime(p), p)) if complete else None


def _repo_from_dir(name):
    if not name.startswith("models--"):
        return None
    org, sep, repo = name[len("models--"):].partition("--")
    return "{}/{}".format(org, repo) if sep and valid_repo("{}/{}".format(org, repo)) else None


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
            if f.endswith(WEIGHT_SUFFIXES):
                weight_files += 1
    return total, weight_files


WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".npz", ".bin")


_SHARD = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})(?P<ext>\.[A-Za-z0-9]+)$")


def _unindexed_weights_complete(snap):
    """No index file: at least one weight file, and every file that names itself `-0000i-of-0000n` has all n
    of its set (the name already says how many shards the checkpoint has)."""
    present = {f for f in os.listdir(snap) if f.endswith(WEIGHT_SUFFIXES) and os.path.isfile(os.path.join(snap, f))}
    for f in present:
        m = _SHARD.match(f)
        if m and any("%s-%05d-of-%s%s" % (m["stem"], i, m["n"], m["ext"]) not in present
                     for i in range(1, int(m["n"]) + 1)):
            return False
    return bool(present)


def snapshot_complete(snap):
    """True when serve could load this snapshot: config.json plus weights. With a safetensors index, every
    shard it names must be a weight file and be there; without one, see _unindexed_weights_complete. An
    interrupted pull can leave the config, the index, or some shards. Malformed index data counts as not
    complete."""
    if not os.path.isfile(os.path.join(snap, "config.json")):
        return False
    index = os.path.join(snap, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return _unindexed_weights_complete(snap)
    try:
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = list(weight_map.values())
    if not all(isinstance(shard, str) and shard.endswith(WEIGHT_SUFFIXES) and os.path.basename(shard) == shard
               for shard in shards):
        return False
    return all(os.path.isfile(os.path.join(snap, shard)) for shard in shards)


def list_local():
    """Every cached repo with a complete snapshot (see snapshot_complete): [{repo, path, bytes, weights}]."""
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
        snap = pick_snapshot(os.path.join(base, name))
        if snap is None:
            continue
        total, weights = _snapshot_info(snap)
        out.append({"repo": repo, "path": snap, "bytes": total, "weights": weights})
    return out


def cached_weight_bytes():
    """{repo: bytes of its .safetensors files} for every cached repo with a complete snapshot: what a rank already
    has on disk and need not download again (the planner's disk check counts it as free)."""
    return {m["repo"]: sum(os.path.getsize(os.path.join(m["path"], f)) for f in os.listdir(m["path"])
                           if f.endswith(".safetensors"))
            for m in list_local()}


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
        raise ModelsError("models pull needs huggingface_hub (pip install huggingface_hub): {}".format(e))
    path = download(repo_id=repo)
    if progress is not None:
        total, weights = _snapshot_info(path)
        progress({"repo": repo, "path": path, "bytes": total, "weights": weights})
    return path
