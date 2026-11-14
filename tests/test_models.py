import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import models


@pytest.fixture
def http_server():
    started = []

    def start(handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, thread))
        return server

    yield start
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def fake_cache(monkeypatch, tmp_path):
    """A HF-cache-shaped tree: one loadable repo, one bare, one junk dir."""
    cache = tmp_path / "hub"
    snap = cache / "models--org--good" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x" * 1000)
    bare = cache / "models--org--nosnap" / "snapshots" / "def456"
    bare.mkdir(parents=True)
    (bare / "readme.md").write_text("hi")
    junk = cache / "datasets--org--set"
    junk.mkdir()
    monkeypatch.setattr(models, "cache_dir", lambda: str(cache))
    return cache


def test_list_local_finds_only_loadable(fake_cache):
    rows = models.list_local()
    assert [r["repo"] for r in rows] == ["org/good"]
    assert rows[0]["weights"] == 1
    assert rows[0]["bytes"] == 1000 + len("{}")
    assert os.path.sep in rows[0]["path"]


def test_list_local_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "cache_dir", lambda: str(tmp_path / "missing"))
    assert models.list_local() == []


def test_pull_downloads_into_cache(fake_cache, monkeypatch):
    seen = {}

    def fake_download(repo_id, **kw):
        seen["repo_id"] = repo_id
        snap = fake_cache / "models--org--good" / "snapshots" / "abc123"
        return str(snap)

    monkeypatch.setattr(models, "_downloader", lambda: fake_download)
    events = []
    path = models.pull("org/good", progress=events.append)
    assert seen["repo_id"] == "org/good"
    assert path.endswith("abc123")
    assert events == [{"repo": "org/good", "path": path, "bytes": 1000 + len("{}"), "weights": 1}]


def test_pull_without_huggingface_hub_exits_cleanly(monkeypatch):
    monkeypatch.setitem(__import__("builtins").__dict__, "__dict__", __import__("builtins").__dict__)
    import builtins
    real_import = builtins.__import__

    def no_hub(name, *a, **k):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    with pytest.raises(models.ModelsError, match="huggingface_hub"):  # library code raises a normal exception, not SystemExit
        models.pull("org/good")


def test_gateway_models_routes(http_server, fake_cache, monkeypatch):
    from omarchy_cluster.gateway import Gateway
    monkeypatch.setattr(models, "cache_dir", lambda: str(fake_cache))
    # bind the real Gateway with its engine pointing nowhere; /models never calls it
    Gateway.engine = "http://127.0.0.1:1"
    server = http_server(Gateway)
    url = "http://127.0.0.1:{}".format(server.server_address[1])
    with urllib.request.urlopen(url + "/models") as r:
        body = json.loads(r.read())
    assert body["data"][0]["repo"] == "org/good"
    req = urllib.request.Request(url + "/models/pull", data=json.dumps({"repo": "org/good"}).encode(),
                                 headers={"Content-Type": "application/json"})
    monkeypatch.setattr(models, "_downloader", lambda: lambda repo_id, **k: str(fake_cache / "models--org--good" / "snapshots" / "abc123"))
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
    assert out["repo"] == "org/good" and out["weights"] == 1
    req = urllib.request.Request(url + "/models/pull", data=json.dumps({"repo": "barename"}).encode())
    try:
        urllib.request.urlopen(req)
        raise AssertionError("bare repo name must be rejected")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def _post(url, body):
    req = urllib.request.Request(url + "/models/pull", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def gateway_url(http_server, fake_cache):
    from omarchy_cluster.gateway import Gateway
    Gateway.engine = "http://127.0.0.1:1"
    return "http://127.0.0.1:{}".format(http_server(Gateway).server_address[1])


@pytest.mark.parametrize("body", [b"[]", b'"x"', b"null", b"3"])
def test_gateway_pull_rejects_a_body_that_is_not_an_object(gateway_url, body):
    status, out = _post(gateway_url, body)
    assert status == 400 and "JSON object" in out["error"]


@pytest.mark.parametrize("repo", ["a/b/c", "org/na me", "../x/y", "org/a--b", "org/", "/name", "org/" + "n" * 100,
                                  "org/..", "org/a..b", "-org/name", "org/-name", ".org/name", "org/.name", "org/name-", "org/name.", "org/name.git"])
def test_gateway_pull_rejects_a_repo_name_huggingface_would_refuse(gateway_url, monkeypatch, repo):
    monkeypatch.setattr(models, "_downloader", lambda: pytest.fail("a bad name must not reach the download"))
    status, out = _post(gateway_url, json.dumps({"repo": repo}).encode())
    assert status == 400 and "org/name" in out["error"]


def test_list_local_skips_a_directory_that_is_not_a_repo_name(fake_cache):
    snap = fake_cache / "models--org--a--b" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert [r["repo"] for r in models.list_local()] == ["org/good"]


def _fake_download_landing(cache, repo="org/new", revision="rev1"):
    """A downloader that behaves like snapshot_download: lands a snapshot with config.json."""
    def download(repo_id, **kw):
        assert repo_id == repo
        snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        (snap / "model.safetensors").write_bytes(b"x" * 1000)
        return str(snap)
    return download


def test_gateway_list_shows_a_pulled_model(http_server, tmp_path, monkeypatch):
    from omarchy_cluster.gateway import Gateway
    cache = tmp_path / "hub"
    monkeypatch.setattr(models, "cache_dir", lambda: str(cache))
    monkeypatch.setattr(models, "_downloader", lambda: _fake_download_landing(cache))
    Gateway.engine = "http://127.0.0.1:1"
    server = http_server(Gateway)
    url = "http://127.0.0.1:{}".format(server.server_address[1])
    assert models.list_local() == []
    status, out = _post(url, json.dumps({"repo": "org/new"}).encode())
    assert status == 200 and out["repo"] == "org/new" and out["weights"] == 1
    with urllib.request.urlopen(url + "/models") as r:
        body = json.loads(r.read())
    assert [row["repo"] for row in body["data"]] == ["org/new"]


def test_pull_restarts_cleanly_after_a_cut_connection(fake_cache, monkeypatch):
    calls = []

    def cut_download(repo_id, **kw):
        calls.append(repo_id)
        if len(calls) == 1:
            blobs = fake_cache / "models--org--new" / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "sha256-partial.incomplete").write_bytes(b"x" * 500)
            _half_pulled(fake_cache, "org--new")  # config, index and shard 1 landed before the cut
            raise ConnectionError("connection cut mid-download")
        snap = fake_cache / "models--org--new" / "snapshots" / "rev1"  # the resume lands the missing shard
        (snap / "model-00002-of-00002.safetensors").write_bytes(b"x" * 100)
        return str(snap)

    monkeypatch.setattr(models, "_downloader", lambda: cut_download)
    with pytest.raises(ConnectionError):
        models.pull("org/new")
    # the half-pulled state is neither listed nor loadable; nothing to clean up by hand
    assert [row["repo"] for row in models.list_local()] == ["org/good"]
    path = models.pull("org/new")
    assert calls == ["org/new", "org/new"]
    assert path.endswith("rev1")
    assert [row["repo"] for row in models.list_local()] == ["org/good", "org/new"]


def test_gateway_answers_502_when_a_pull_fails(gateway_url, monkeypatch):
    def cut(repo_id, **kw):
        raise ConnectionError("connection cut mid-download")

    monkeypatch.setattr(models, "_downloader", lambda: cut)
    status, out = _post(gateway_url, json.dumps({"repo": "org/good"}).encode())
    assert status == 502 and "pull:" in out["error"]


def test_a_pulled_model_resolves_for_place_and_serve(tmp_path, monkeypatch):
    from omarchy_cluster import planner
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(models, "_downloader", lambda: _fake_download_landing(
        tmp_path / ".cache/huggingface/hub"))
    path = models.pull("org/new", progress=lambda info: None)
    assert planner.find_model_path("org/new") == path
    assert planner.model_info("org/new")["weights_bytes"] == 1000


def test_place_names_an_unpulled_model_as_missing(tmp_path, monkeypatch):
    from omarchy_cluster import cli
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli.discover, "discover_nodes", lambda *a, **k: {"m1": {"facts": {}}})
    monkeypatch.setattr(cli.probe_mod, "load_links", lambda path: {"pairs": []})
    args = argparse.Namespace(model="org/unpulled", nodes=None, links="links.json")
    with pytest.raises(SystemExit, match=r"model not found locally or in HF cache: org/unpulled"):
        cli.cmd_place(args)


def _half_pulled(cache, repo="org--partial"):
    """An interrupted multi-file pull: config, the shard index and shard 1 landed; shard 2 is only a .incomplete blob."""
    root = cache / ("models--" + repo)
    snap = root / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (root / "refs").mkdir(exist_ok=True)
    (root / "refs" / "main").write_text("rev1")
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "hidden_size": 8,
                                                  "num_attention_heads": 2}))
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
    (root / "blobs").mkdir(exist_ok=True)
    (root / "blobs" / "sha256-two.incomplete").write_bytes(b"x" * 10)
    return snap


def test_a_snapshot_missing_a_shard_is_not_listed(fake_cache):
    _half_pulled(fake_cache)
    assert [row["repo"] for row in models.list_local()] == ["org/good"]


def test_a_snapshot_with_every_shard_is_listed(fake_cache):
    snap = _half_pulled(fake_cache)
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"x" * 100)
    assert [row["repo"] for row in models.list_local()] == ["org/good", "org/partial"]


def test_the_planner_does_not_resolve_or_size_a_half_pulled_model(fake_cache, monkeypatch, tmp_path):
    from omarchy_cluster import planner
    home = tmp_path / "home"
    target = home / ".cache" / "huggingface"
    target.mkdir(parents=True)
    os.symlink(str(fake_cache), str(target / "hub"))
    monkeypatch.setenv("HOME", str(home))
    snap = _half_pulled(fake_cache)
    assert planner.find_model_path("org/partial") is None
    assert planner.model_info("org/partial") is None
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"x" * 100)
    assert planner.find_model_path("org/partial") == str(fake_cache / "models--org--partial" / "snapshots" / "rev1")  # the planner reads the same cache_dir as models



def _snap_with_config(cache, repo="org--cfg", index=None, weights=()):
    snap = cache / ("models--" + repo) / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    if index is not None:
        (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    for name in weights:
        (snap / name).write_bytes(b"x" * 10)
    return snap


def test_a_config_only_snapshot_is_not_listed_or_resolved(fake_cache):
    snap = _snap_with_config(fake_cache)
    assert not models.snapshot_complete(str(snap))
    assert [row["repo"] for row in models.list_local()] == ["org/good"]


def test_an_unindexed_single_file_model_is_listed(fake_cache):
    _snap_with_config(fake_cache, weights=["model.safetensors"])
    assert [row["repo"] for row in models.list_local()] == ["org/cfg", "org/good"]


@pytest.mark.parametrize("index", [
    {"weight_map": {"a": []}},
    {"weight_map": {"a": 1}},
    {"weight_map": {}},
    {"weight_map": []},
    {"weight_map": {"a": "../outside.safetensors"}},
    {"weight_map": {"a": "config.json"}},
    {"weight_map": {"a": "tokenizer.json", "b": "model.safetensors"}},
    {"no": "weight_map"},
    [],
])
def test_a_malformed_or_empty_index_excludes_the_snapshot_without_raising(fake_cache, index):
    _snap_with_config(fake_cache, index=index, weights=["model.safetensors"])
    assert [row["repo"] for row in models.list_local()] == ["org/good"]


def test_one_shard_of_a_two_shard_checkpoint_with_no_index_is_not_listed(fake_cache):
    snap = _snap_with_config(fake_cache, weights=["model-00001-of-00002.safetensors"])
    assert not models.snapshot_complete(str(snap))
    assert [row["repo"] for row in models.list_local()] == ["org/good"]


def test_every_shard_of_an_unindexed_sharded_checkpoint_makes_it_listed(fake_cache):
    _snap_with_config(fake_cache, weights=["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    assert [row["repo"] for row in models.list_local()] == ["org/cfg", "org/good"]


def test_an_unindexed_gguf_set_is_judged_by_its_own_shard_count(fake_cache):
    snap = _snap_with_config(fake_cache, weights=["m-00001-of-00003.gguf", "m-00003-of-00003.gguf"])
    assert not models.snapshot_complete(str(snap))


def _snapshot(cache, repo, revision, mtime, weights=b"w"):
    snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(weights)
    os.utime(snap, (mtime, mtime))
    return snap


def test_the_cache_dir_follows_the_huggingface_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    assert models.cache_dir() == str(tmp_path / ".cache/huggingface/hub")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert models.cache_dir() == str(tmp_path / "hfhome/hub")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "moved"))
    assert models.cache_dir() == str(tmp_path / "moved")


def test_list_local_finds_a_model_in_a_moved_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "moved"))
    _snapshot(tmp_path / "moved", "org/good", "aaa", 100)
    assert [m["repo"] for m in models.list_local()] == ["org/good"]


def test_two_snapshots_of_one_repo_follow_refs_main(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    old = _snapshot(tmp_path, "org/good", "aaa-old", 100)
    new = _snapshot(tmp_path, "org/good", "zzz-new", 200)
    (tmp_path / "models--org--good/refs").mkdir()
    (tmp_path / "models--org--good/refs/main").write_text("aaa-old\n")  # main still points at the older one
    assert [m["path"] for m in models.list_local()] == [str(old)]
    (tmp_path / "models--org--good/refs/main").write_text("zzz-new\n")
    assert [m["path"] for m in models.list_local()] == [str(new)]


def test_without_refs_main_the_newest_complete_snapshot_wins_not_the_first_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _snapshot(tmp_path, "org/good", "aaa-old", 100)
    new = _snapshot(tmp_path, "org/good", "zzz-new", 200)
    _snapshot(tmp_path, "org/good", "mmm-newest-but-partial", 300)
    (tmp_path / "models--org--good/snapshots/mmm-newest-but-partial/model.safetensors").unlink()
    assert [m["path"] for m in models.list_local()] == [str(new)]


def test_the_planner_finds_a_model_in_the_moved_cache_too(tmp_path, monkeypatch):
    from omarchy_cluster import planner
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snap = _snapshot(tmp_path, "org/good", "aaa", 100)
    assert planner.find_model_path("org/good") == str(snap)


def _json_only_snapshot(tmp_path, total=17_200_000_000):
    snap = tmp_path / "json-only"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 48, "hidden_size": 2048,
                                                  "num_key_value_heads": 4, "num_attention_heads": 32}))
    (snap / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total}, "weight_map": {"a": "model-00001-of-00002.safetensors",
                                                         "b": "model-00002-of-00002.safetensors"}}))
    return snap


def test_a_model_is_sized_from_its_index_when_this_rank_holds_no_weight_files(tmp_path):
    """Issue 96: a rank that holds only its own shards (or none yet) sized the model from the files present."""
    from omarchy_cluster import planner
    snap = _json_only_snapshot(tmp_path)
    info = planner.model_info_from_index(str(snap), "org/big")
    assert (info["layers"], info["weights_bytes"], info["ref"]) == (48, 17_200_000_000, "org/big")


def test_an_index_without_a_total_falls_back_to_the_files_present(tmp_path):
    from omarchy_cluster import planner
    snap = _json_only_snapshot(tmp_path)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (snap / "model.safetensors").write_bytes(b"x" * 10)
    assert planner.model_info_from_index(str(snap), "org/big")["weights_bytes"] == 10


def test_the_split_sizes_an_uncached_model_from_the_json_only_download(tmp_path):
    from omarchy_cluster import rank
    snap = _json_only_snapshot(tmp_path)
    asked = []
    info = rank._split_info("org/not-here", lambda ref: asked.append(ref) or str(snap))
    assert asked == ["org/not-here"] and info["weights_bytes"] == 17_200_000_000


def test_the_split_does_not_download_anything_for_a_model_held_whole(tmp_path):
    from omarchy_cluster import rank
    snap = _json_only_snapshot(tmp_path)
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"x")
    info = rank._split_info(str(snap), lambda ref: pytest.fail("downloaded"))
    assert info["path"] == str(snap)


def test_layers_are_always_timed_from_a_config_only_copy_so_every_rank_times_the_same_thing(tmp_path):
    """Random-weight layers time differently from real ones (MoE routing), so a rank that holds the weights must not
    time those while its peer times random ones: the copy holds the json files and no weight file."""
    from omarchy_cluster import rank
    snap = _json_only_snapshot(tmp_path)
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (snap / "notes.txt").write_text("not json")
    copy = rank._config_only_dir(str(snap))
    try:
        assert sorted(os.listdir(copy)) == ["config.json", "model.safetensors.index.json"]
    finally:
        import shutil
        shutil.rmtree(copy)


def test_the_split_sizes_a_local_partial_or_json_only_dir_from_the_index_not_the_files_present(tmp_path):
    """A local path exists but is not held whole (its own shards, or json only): the files present under-report,
    and the split's layer cap would be wrong. The index total decides (issue 96, review)."""
    from omarchy_cluster import rank
    for present in (0, 1_000_000_000):
        case = tmp_path / ("case-%d" % present)
        case.mkdir()
        snap = _json_only_snapshot(case)
        if present:
            (snap / "model-00001-of-00002.safetensors").write_bytes(b"x" * present)
        info = rank._split_info(str(snap), lambda ref: pytest.fail("downloaded"))
        assert (info["weights_bytes"], info["layers"], info["path"]) == (17_200_000_000, 48, str(snap))
