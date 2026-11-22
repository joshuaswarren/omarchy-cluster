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


def test_kv_put_refuses_a_short_body(tmp_path):
    """A body shorter than Content-Length must not publish a truncated slot file."""
    path = os.path.join(str(tmp_path), "slot.bin")
    with pytest.raises(ValueError):
        agent.kv_put(path, io.BytesIO(b"short"), 100)
    assert not os.path.exists(path)
    assert [f for f in os.listdir(str(tmp_path)) if f.startswith(".kv-")] == []


def test_kv_post_short_body_gets_400_and_writes_nothing(kv_agent):
    """End to end: a dropped connection mid-body is a 400, not a 200 + short file."""
    import socket
    url, kvdir = kv_agent
    with socket.create_connection(("127.0.0.1", int(url.rpartition(":")[2]))) as s:
        s.sendall(b"POST /v1/kv?name=x.bin HTTP/1.1\r\nHost: x\r\n"
                  b"X-Cluster-Token: " + TOKEN.encode() +
                  b"\r\nContent-Length: 100\r\nContent-Type: application/octet-stream\r\n\r\nshort")
        s.shutdown(socket.SHUT_WR)
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += s.recv(4096)
    assert b" 400 " in resp.split(b"\r\n", 1)[0]
    assert os.listdir(kvdir) == []


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


def test_agent_push_refuses_a_short_write_reply(monkeypatch, tmp_path):
    """The engine must not report a full transfer when the agent wrote less."""
    from omarchy_cluster import joint_engine as je
    src = tmp_path / "s.bin"
    src.write_bytes(b"12345")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"bytes": 3}).encode()

    monkeypatch.setattr(je.client, "urlopen", lambda req, timeout, pin=None: _Resp())
    monkeypatch.setattr(je.client, "agent_url", lambda *a, **k: "https://x/v1/kv")
    monkeypatch.setattr(je.client, "_headers", lambda extra=None, token=None: {})
    with pytest.raises(je.JointError):
        je._agent_push("10.0.0.9", 8025, "s.bin", str(src))


def test_agent_push_accepts_a_matching_byte_count(monkeypatch, tmp_path):
    from omarchy_cluster import joint_engine as je
    src = tmp_path / "s.bin"
    src.write_bytes(b"12345")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"bytes": 5}).encode()

    monkeypatch.setattr(je.client, "urlopen", lambda req, timeout, pin=None: _Resp())
    monkeypatch.setattr(je.client, "agent_url", lambda *a, **k: "https://x/v1/kv")
    monkeypatch.setattr(je.client, "_headers", lambda extra=None, token=None: {})
    assert je._agent_push("10.0.0.9", 8025, "s.bin", str(src)) == 5
