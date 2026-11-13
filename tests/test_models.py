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
    with pytest.raises(SystemExit, match="huggingface_hub"):
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
            raise ConnectionError("connection cut mid-download")
        return _fake_download_landing(fake_cache)(repo_id)

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
