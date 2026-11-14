import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import agent, bootstrap, client
from omarchy_cluster import token as cluster_token

TOKEN = "t0k3n-for-the-test"
ACTIONS = ("rank_start", "rank_stop", "rpc_start", "gpu_cap", "tcp_rtt", "udp_rtt", "blast", "sink_serve",
           "iperf_server", "iperf_client")
GET_ROUTES = ["/v1/health", "/v1/facts", "/v1/nothing"]
POST_ROUTES = ["/v1/rank/start", "/v1/rank/stop", "/v1/rpc/start", "/v1/gpu-cap", "/v1/rtt", "/v1/blast",
               "/v1/sink-serve", "/v1/iperf-server", "/v1/iperf-client", "/v1/nothing"]


@pytest.fixture
def agent_url(monkeypatch):
    monkeypatch.setattr(cluster_token, "read", lambda: TOKEN)
    ran = []
    for name in ACTIONS:
        monkeypatch.setattr(agent, name, lambda *a, _n=name, **k: ran.append(_n))
    hb = SimpleNamespace(seq=3, age=lambda: 0.5)
    monkeypatch.setattr(agent.Handler, "agent", SimpleNamespace(hb=hb, facts=lambda: {"hostname": "h"}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), agent.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % server.server_port, ran
    finally:
        server.shutdown()
        server.server_close()


def _status(url, path, method, token):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Cluster-Token"] = token
    req = urllib.request.Request(url + path, data=b"{}" if method == "POST" else None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


@pytest.mark.parametrize("token", [None, "wrong"])
@pytest.mark.parametrize("path", GET_ROUTES)
def test_every_get_route_needs_the_cluster_token(agent_url, path, token):
    url, ran = agent_url
    assert _status(url, path, "GET", token) == 403
    assert ran == []


@pytest.mark.parametrize("token", [None, "wrong"])
@pytest.mark.parametrize("path", POST_ROUTES)
def test_every_post_route_needs_the_cluster_token_and_runs_nothing_without_it(agent_url, path, token):
    url, ran = agent_url
    assert _status(url, path, "POST", token) == 403
    assert ran == []


@pytest.mark.parametrize("path", ["/v1/health", "/v1/facts"])
def test_the_token_opens_the_get_routes(agent_url, path):
    url, _ = agent_url
    assert _status(url, path, "GET", TOKEN) == 200


def test_the_token_still_reaches_a_post_route(agent_url):
    url, ran = agent_url
    assert _status(url, "/v1/rtt", "POST", TOKEN) == 400  # authenticated, then the missing "ip" field
    assert ran == []


def _fake_agent(seen):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("X-Cluster-Token")))
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET
    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setattr(cluster_token, "read", lambda: TOKEN)
    seen = []
    server = _fake_agent(seen)
    try:
        yield server.server_port, seen
    finally:
        server.shutdown()
        server.server_close()


def test_the_clients_send_the_local_token_to_the_agent(fake_agent):
    port, seen = fake_agent
    client.fetch_facts("127.0.0.1", port)
    client.post("127.0.0.1", port, "/v1/rtt", {})
    assert bootstrap.agent_answers("127.0.0.1", port, wait_s=1)
    assert seen == [("/v1/facts", TOKEN), ("/v1/rtt", TOKEN), ("/v1/health", TOKEN)]
