import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster.gateway import Gateway


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


def stub_engine(routes):
    class StubEngine(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            code, body = routes.get(self.path, (404, b'{"error": "not found"}'))
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return StubEngine


def base_url(server):
    return f"http://127.0.0.1:{server.server_port}"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("body", [
    b"[]", b"null", b"42", b"{",
    b'{"max_tokens": null}', b'{"max_tokens": "invalid"}',
    b'{"max_tokens": []}', b'{"max_tokens": 1e1000}',
])
def test_invalid_completion_request_returns_http_400(path, body, monkeypatch, http_server):
    server = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(server))
    request = urllib.request.Request(
        base_url(server) + path,
        data=body, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=2)
    with error.value as response:
        assert response.code == 400
        assert isinstance(json.load(response)["error"], str)


@pytest.mark.parametrize("path", ["/health", "/v1/health"])
def test_health_returns_ok_when_engine_is_up(path, monkeypatch, http_server):
    engine = http_server(stub_engine({"/health": (200, b"{}")}))
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    gateway = http_server(Gateway)
    with urllib.request.urlopen(base_url(gateway) + path, timeout=2) as response:
        assert response.code == 200
        assert json.load(response) == {"ok": True}


def test_health_returns_503_when_engine_errors(monkeypatch, http_server):
    engine = http_server(stub_engine({"/health": (500, b"engine exploded")}))
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    gateway = http_server(Gateway)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base_url(gateway) + "/health", timeout=2)
    with error.value as response:
        assert response.code == 503
        assert json.load(response) == {"ok": False}


@pytest.mark.parametrize("path", ["/status", "/v1/status"])
def test_status_returns_502_when_engine_errors(path, monkeypatch, http_server):
    engine = http_server(stub_engine({"/status": (500, b"engine exploded")}))
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    gateway = http_server(Gateway)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base_url(gateway) + path, timeout=2)
    with error.value as response:
        assert response.code == 502
        assert json.load(response)["error"].startswith("engine: ")


def test_unknown_get_route_returns_404(http_server):
    gateway = http_server(Gateway)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base_url(gateway) + "/v1/nope", timeout=2)
    with error.value as response:
        assert response.code == 404
        assert isinstance(json.load(response)["error"], str)


def test_unknown_post_route_returns_404(http_server):
    gateway = http_server(Gateway)
    request = urllib.request.Request(
        base_url(gateway) + "/v1/nope", data=b"{}",
        headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=2)
    with error.value as response:
        assert response.code == 404
        assert isinstance(json.load(response)["error"], str)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_nonzero_temperature_is_rejected_before_the_engine(path, monkeypatch, http_server):
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(gateway))
    request = urllib.request.Request(
        base_url(gateway) + path,
        data=json.dumps({"temperature": 0.7}).encode(),
        headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=2)
    with error.value as response:
        assert response.code == 400
        assert "temperature" in json.load(response)["error"]