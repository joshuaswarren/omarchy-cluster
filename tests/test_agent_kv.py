"""The agent's scoped /v1/kv endpoint: basename-validated slot-file transport."""
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import agent
from omarchy_cluster import token as cluster_token

TOKEN = "kv-token"


@pytest.fixture
def kv_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_token, "read", lambda: TOKEN)
    monkeypatch.setattr(agent, "KV_DIR", str(tmp_path))
    hb = SimpleNamespace(seq=0, age=lambda: 0.0)
    monkeypatch.setattr(agent.Handler, "agent", SimpleNamespace(hb=hb, facts=lambda: {}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), agent.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % server.server_port, str(tmp_path)
    finally:
        server.shutdown()
        server.server_close()


def _req(url, method, path, body=None, token=TOKEN):
    headers = {}
    if token is not None:
        headers["X-Cluster-Token"] = token
    req = urllib.request.Request(url + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.mark.parametrize("name", ["", "..", "../x", "a/b", "/etc/passwd", "a\\b",
                                  "x" * 129, "na me", "n;rm", ".h/../x"])
def test_kv_names_must_be_plain_basenames(name, tmp_path):
    with pytest.raises(ValueError):
        agent.kv_file_path(name, str(tmp_path))


def test_kv_file_path_accepts_a_basename(tmp_path):
    assert agent.kv_file_path("joint-kv.bin", str(tmp_path)) == str(tmp_path) + "/joint-kv.bin"
    assert agent.kv_file_path("slot_0-AB.CD", str(tmp_path)).endswith("/slot_0-AB.CD")


def test_kv_put_streams_and_renames(tmp_path):
    path = os.path.join(str(tmp_path), "slot.bin")
    body = os.urandom(3 << 20)  # multi-chunk
    assert agent.kv_put(path, io.BytesIO(body), len(body)) == len(body)
    assert open(path, "rb").read() == body
    assert [f for f in os.listdir(str(tmp_path)) if f.startswith(".kv-")] == []


def test_kv_put_cleans_the_temp_file_on_short_body(tmp_path):
    path = os.path.join(str(tmp_path), "slot.bin")
    assert agent.kv_put(path, io.BytesIO(b"short"), 100) == 5
    assert open(path, "rb").read() == b"short"
    assert [f for f in os.listdir(str(tmp_path)) if f.startswith(".kv-")] == []


def test_kv_post_then_get_roundtrip(kv_agent):
    url, kvdir = kv_agent
    body = b"kv-" + os.urandom(1024)
    code, reply = _req(url, "POST", "/v1/kv?name=slot0.bin", body)
    assert code == 200 and json.loads(reply)["bytes"] == len(body)
    assert open(os.path.join(kvdir, "slot0.bin"), "rb").read() == body
    code, got = _req(url, "GET", "/v1/kv?name=slot0.bin")
    assert code == 200 and got == body


@pytest.mark.parametrize("name", ["..", "a/b", "/etc/passwd", "x" * 129, ""])
def test_kv_get_rejects_bad_names(kv_agent, name):
    url, _ = kv_agent
    code, _ = _req(url, "GET", "/v1/kv?name=" + urllib.parse.quote(name, safe=""))
    assert code == 400


@pytest.mark.parametrize("name", ["..", "a/b"])
def test_kv_post_rejects_bad_names(kv_agent, name):
    url, kvdir = kv_agent
    code, _ = _req(url, "POST", "/v1/kv?name=" + urllib.parse.quote(name, safe=""), b"x")
    assert code == 400
    assert os.listdir(kvdir) == []


def test_kv_get_missing_is_404(kv_agent):
    url, _ = kv_agent
    code, _ = _req(url, "GET", "/v1/kv?name=nope.bin")
    assert code == 404


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_kv_needs_the_token(kv_agent, method):
    url, kvdir = kv_agent
    code, _ = _req(url, method, "/v1/kv?name=slot.bin", b"x" if method == "POST" else None, token=None)
    assert code == 403
    code, _ = _req(url, method, "/v1/kv?name=slot.bin", b"x" if method == "POST" else None, token="wrong")
    assert code == 403
    assert os.listdir(kvdir) == []
