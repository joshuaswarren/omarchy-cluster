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


@pytest.mark.parametrize("repo", ["a/b/c", "org/na me", "../x/y", "org/a--b", "org/", "/name", "org/" + "n" * 100])
def test_gateway_pull_rejects_a_repo_name_huggingface_would_refuse(gateway_url, monkeypatch, repo):
    monkeypatch.setattr(models, "_downloader", lambda: pytest.fail("a bad name must not reach the download"))
    status, out = _post(gateway_url, json.dumps({"repo": repo}).encode())
    assert status == 400 and "org/name" in out["error"]


def test_list_local_skips_a_directory_that_is_not_a_repo_name(fake_cache):
    snap = fake_cache / "models--org--a--b" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert [r["repo"] for r in models.list_local()] == ["org/good"]
