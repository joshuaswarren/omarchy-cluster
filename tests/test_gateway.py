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


ENGINE_REPLY = {"text": "x", "prompt_tokens": 1, "completion_tokens": 1, "prefill_ms": 1.0,
                "decode_ms": 1.0, "tokps": 1.0}


def recording_engine(seen):
    class Engine(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(ENGINE_REPLY).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Engine


def post(gateway, path, body):
    request = urllib.request.Request(base_url(gateway) + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=2)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_sampling_settings_reach_the_engine(path, monkeypatch, http_server):
    seen = []
    engine = http_server(recording_engine(seen))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    post(gateway, path, {"temperature": 0.7, "top_p": 0.9, "seed": 5}).close()
    assert seen[0]["greedy"] is False
    assert seen[0]["sampling"] == {"temperature": 0.7, "top_p": 0.9, "seed": 5}


def test_temperature_zero_stays_greedy_without_sampling(monkeypatch, http_server):
    seen = []
    engine = http_server(recording_engine(seen))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    post(gateway, "/v1/completions", {"temperature": 0}).close()
    assert seen[0]["greedy"] is True and "sampling" not in seen[0]


@pytest.mark.parametrize("body", [{"temperature": -1}, {"temperature": "hot"}, {"temperature": 0.5, "top_p": 0},
                                  {"temperature": 0.5, "top_p": 1.5}, {"temperature": 0.5, "seed": "x"},
                                  {"temperature": 0.5, "seed": -1}, {"temperature": 0.5, "seed": 2 ** 32}])
def test_invalid_sampling_is_rejected_before_the_engine(body, monkeypatch, http_server):
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", "http://127.0.0.1:9")
    with pytest.raises(urllib.error.HTTPError) as error:
        post(gateway, "/v1/completions", body)
    with error.value as response:
        assert response.code == 400


@pytest.fixture
def remote_peer(monkeypatch, tmp_path):
    """Every request reaches the gateway as if it came from another machine."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Gateway, "_peer_is_local", lambda self: False)
    return tmp_path


def write_token(home, token):
    path = home / ".config" / "omarchy-cluster" / "token"
    path.parent.mkdir(parents=True)
    path.write_text(token + "\n")


def remote_request(gateway, path, bearer=None, data=None):
    headers = {"Content-Type": "application/json"}
    if bearer is not None:
        headers["Authorization"] = "Bearer " + bearer
    return urllib.request.Request(base_url(gateway) + path, data=data, headers=headers)


@pytest.mark.parametrize("path,data", [
    ("/v1/models", None),
    ("/v1/status", None),
    ("/models", None),
    ("/v1/chat/completions", b"{}"),
    ("/models/pull", b'{"repo": "org/name"}'),
])
def test_remote_request_without_a_token_is_refused(path, data, remote_peer, http_server):
    write_token(remote_peer, "secret-token")
    gateway = http_server(Gateway)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(remote_request(gateway, path, data=data), timeout=2)
    with error.value as response:
        assert response.code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert isinstance(json.load(response)["error"], str)


def test_remote_request_with_a_wrong_token_is_refused(remote_peer, http_server):
    write_token(remote_peer, "secret-token")
    gateway = http_server(Gateway)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(remote_request(gateway, "/v1/models", bearer="guess"), timeout=2)
    assert error.value.code == 401
    error.value.close()


def test_remote_request_is_refused_when_the_node_has_no_token(remote_peer, http_server):
    gateway = http_server(Gateway)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(remote_request(gateway, "/v1/models", bearer=""), timeout=2)
    assert error.value.code == 401
    error.value.close()


def test_remote_request_with_the_cluster_token_is_served(remote_peer, http_server):
    write_token(remote_peer, "secret-token")
    gateway = http_server(Gateway)
    with urllib.request.urlopen(
            remote_request(gateway, "/v1/models", bearer="secret-token"), timeout=2) as response:
        assert response.code == 200
        assert json.load(response)["data"][0]["id"] == "omarchy-cluster"


def test_health_needs_no_token_from_a_remote_peer(remote_peer, monkeypatch, http_server):
    engine = http_server(stub_engine({"/health": (200, b"{}")}))
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    gateway = http_server(Gateway)
    with urllib.request.urlopen(remote_request(gateway, "/health"), timeout=2) as response:
        assert response.code == 200


class GateEngine:
    """Engine stub that counts requests inside /generate and holds each until released."""

    def __init__(self):
        self.lock = threading.Lock()
        self.inside = 0
        self.peak = 0
        self.release = threading.Event()
        self.entered = threading.Semaphore(0)

    def handler(self):
        gate = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                with gate.lock:
                    gate.inside += 1
                    gate.peak = max(gate.peak, gate.inside)
                gate.entered.release()
                gate.release.wait(10)
                with gate.lock:
                    gate.inside -= 1
                body = json.dumps({"text": "ok", "prompt_tokens": 1, "completion_tokens": 1,
                                   "prefill_ms": 0.0, "decode_ms": 0.0, "tokps": 1.0}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def post_completion(gateway, results):
    request = urllib.request.Request(
        base_url(gateway) + "/v1/completions", data=b'{"prompt": "hi"}',
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            results.append((response.code, None))
    except urllib.error.HTTPError as e:
        results.append((e.code, e.headers.get("Retry-After")))
        e.close()


def run_clients(gateway, count):
    results = []
    threads = [threading.Thread(target=post_completion, args=(gateway, results)) for _ in range(count)]
    for t in threads:
        t.start()
    return threads, results


def test_gateway_runs_as_many_requests_at_once_as_it_has_slots(monkeypatch, http_server):
    gate = GateEngine()
    engine = http_server(gate.handler())
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    monkeypatch.setattr(Gateway, "slots", threading.BoundedSemaphore(2))
    gateway = http_server(Gateway)
    threads, results = run_clients(gateway, 4)
    for _ in range(2):
        assert gate.entered.acquire(timeout=5)
    assert not gate.entered.acquire(timeout=0.3)  # the other two wait at the gateway
    assert gate.inside == 2
    gate.release.set()
    for t in threads:
        t.join(timeout=10)
    assert sorted(code for code, _ in results) == [200] * 4
    assert gate.peak == 2


def test_gateway_answers_503_with_retry_after_when_the_wait_for_a_slot_runs_out(monkeypatch, http_server):
    gate = GateEngine()
    engine = http_server(gate.handler())
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    monkeypatch.setattr(Gateway, "slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(Gateway, "queue_wait_s", 0.3)
    gateway = http_server(Gateway)
    first, first_results = run_clients(gateway, 1)
    assert gate.entered.acquire(timeout=5)
    late, late_results = run_clients(gateway, 1)
    late[0].join(timeout=10)
    assert late_results == [(503, "5")]
    gate.release.set()
    first[0].join(timeout=10)
    assert first_results == [(200, None)]
    assert gate.peak == 1


def test_a_failed_engine_call_frees_its_slot(monkeypatch, http_server):
    monkeypatch.setattr(Gateway, "engine", "http://127.0.0.1:1")
    monkeypatch.setattr(Gateway, "slots", threading.BoundedSemaphore(1))
    gateway = http_server(Gateway)
    for _ in range(2):
        results = []
        post_completion(gateway, results)
        assert results == [(502, None)]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_serve_rejects_a_slot_count_below_one(value, capsys):
    from omarchy_cluster import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--model", "m", "--max-concurrent", value])
    assert e.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


def refusing_engine(code, reply):
    class Engine(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(reply).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Engine


@pytest.mark.parametrize("code,reply", [
    (400, {"error": "ranks sample greedily"}),
    (200, {"error": "sushi is greedy by format", "finish_reason": "error", "client_error": True}),
])
def test_an_engine_refusal_reaches_the_client_as_a_400_with_its_text(code, reply, monkeypatch, http_server):
    engine = http_server(refusing_engine(code, reply))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    with pytest.raises(urllib.error.HTTPError) as error:
        post(gateway, "/v1/completions", {"temperature": 0.7}).close()
    with error.value as response:
        assert response.code == 400
        assert json.load(response)["error"] == reply["error"]
