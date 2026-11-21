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


def status_engine(status):
    class Engine(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            body = json.dumps(status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(ENGINE_REPLY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Engine


@pytest.mark.parametrize("asked,loaded", [
    ("mlx-community/other", "mlx-community/mini"),
    ("mlx-community/mini", "mlx-community/other"),
    ("xmini", "mlx-community/mini"),
])
def test_a_request_for_another_model_is_refused_with_both_names(asked, loaded, monkeypatch, http_server):
    engine = http_server(status_engine({"ok": True, "model": loaded}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    with pytest.raises(urllib.error.HTTPError) as error:
        post(gateway, "/v1/chat/completions",
             {"messages": [{"role": "user", "content": "hi"}], "model": asked})
    with error.value as response:
        assert response.code == 400
        message = json.load(response)["error"]
        assert loaded in message and asked in message


@pytest.mark.parametrize("asked,loaded", [
    (None, "mlx-community/mini"),
    ("omarchy-cluster", "mlx-community/mini"),
    ("mlx-community/mini", "mlx-community/mini"),
    ("mini", "mlx-community/mini"),
])
def test_a_matching_or_absent_model_reaches_the_engine(asked, loaded, monkeypatch, http_server):
    engine = http_server(status_engine({"ok": True, "model": loaded}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    if asked is not None:
        body["model"] = asked
    with post(gateway, "/v1/chat/completions", body) as response:
        assert response.code == 200


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


def sse_engine(chunks, final):
    """An engine whose /generate answers chunked SSE: chunk events then the final event."""

    class Engine(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not req.get("stream"):
                body = json.dumps({"text": "".join(chunks) if chunks else final.get("text", ""),
                                   "prompt_tokens": final.get("prompt_tokens", 0),
                                   "completion_tokens": final.get("completion_tokens", 0),
                                   "prefill_ms": final.get("prefill_ms", 0.0),
                                   "decode_ms": final.get("decode_ms", 0.0),
                                   "tokps": final.get("tokps", 0.0)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for text in chunks:
                self.wfile.write(("data: %s\n\n" % json.dumps({"chunk": text})).encode())
                self.wfile.flush()
            self.wfile.write(("data: %s\n\n" % json.dumps(final)).encode())
            self.wfile.flush()

    return Engine


def test_a_stream_request_gets_sse_deltas_in_order(monkeypatch, http_server):
    """The issue-95 contract: stream=true must stream. This test fails against a gateway
    that drops the stream key and answers one JSON body (no data: events arrive)."""
    engine = http_server(sse_engine(["Toke", "nize", " me"], {
        "text": "Tokenize me", "prompt_tokens": 3, "completion_tokens": 3,
        "prefill_ms": 1.0, "decode_ms": 2.0, "tokps": 1.5}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    request = urllib.request.Request(
        base_url(gateway) + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=2) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        events = []
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        deltas = [e["choices"][0]["delta"]["content"] for e in events if "choices" in e]
        assert deltas == ["Toke", "nize", " me"]
        finals = [e for e in events if "choices" not in e]
        assert finals and finals[0]["usage"]["completion_tokens"] == 3
        assert finals[0]["timings"]["tokens_per_s"] == 1.5


def test_a_stream_request_still_sends_done_last(monkeypatch, http_server):
    engine = http_server(sse_engine(["a"], {"text": "a", "prompt_tokens": 1,
                                            "completion_tokens": 1, "prefill_ms": 1.0,
                                            "decode_ms": 1.0, "tokps": 1.0}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    request = urllib.request.Request(
        base_url(gateway) + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=2) as r:
        lines = [raw.decode(errors="replace").strip() for raw in r if raw.strip()]
    assert lines[-1] == "data: [DONE]"


def test_a_non_stream_request_stays_one_json_body(monkeypatch, http_server):
    engine = http_server(sse_engine(["x"], {
        "text": "x", "prompt_tokens": 1, "completion_tokens": 1,
        "prefill_ms": 1.0, "decode_ms": 1.0, "tokps": 1.0}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    with post(gateway, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}) as r:
        assert not r.headers["Content-Type"].startswith("text/event-stream")
        body = json.load(r)
    assert body["choices"][0]["message"]["content"] == "x"


def json_engine(reply):
    """A JSON-only engine: it ignores `stream` and answers /generate with one JSON
    body — the llama.cpp and sushi reply shape (issue 120)."""

    class Engine(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Engine


def _stream_lines(gateway):
    request = urllib.request.Request(
        base_url(gateway) + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=2) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        return [raw.decode(errors="replace").strip() for raw in r if raw.strip()]


def test_stream_over_a_json_only_engine_delivers_the_completion(monkeypatch, http_server):
    """llama.cpp and sushi answer /generate with one JSON object even when stream=true
    (issue 120): the gateway must still emit the text, the usage event and [DONE]."""
    engine = http_server(json_engine({
        "text": "sushi says hi", "prompt_tokens": 4, "completion_tokens": 3,
        "prefill_ms": 1.0, "decode_ms": 2.0, "tokps": 1.5, "finish_reason": "stop"}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    lines = _stream_lines(gateway)
    events = [json.loads(line[6:]) for line in lines
              if line.startswith("data: ") and line != "data: [DONE]"]
    deltas = [e["choices"][0]["delta"]["content"] for e in events if "choices" in e]
    assert deltas == ["sushi says hi"]
    finals = [e for e in events if "choices" not in e]
    assert finals and finals[0]["usage"]["completion_tokens"] == 3
    assert lines[-1] == "data: [DONE]"


def test_stream_over_a_json_only_engine_relays_an_engine_error(monkeypatch, http_server):
    """A JSON error reply on a streamed request becomes the error event, then [DONE]."""
    engine = http_server(json_engine({"error": "llama-server: out of slots"}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    lines = _stream_lines(gateway)
    events = [json.loads(line[6:]) for line in lines
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert events[0]["error"] == "llama-server: out of slots"
    assert lines[-1] == "data: [DONE]"


def test_stream_over_a_json_only_engine_names_the_client_error(monkeypatch, http_server):
    """A client_error reply without an error string still names the engine and the
    cause class instead of the generic 'engine error' (PR 129 follow-up)."""
    engine = http_server(json_engine({"client_error": True}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    lines = _stream_lines(gateway)
    events = [json.loads(line[6:]) for line in lines
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert events[0]["error"] == "%s: client error" % base_url(engine)


def test_stream_over_a_json_only_engine_uses_a_client_error_string(monkeypatch, http_server):
    """A string client_error is itself the useful message."""
    engine = http_server(json_engine({"client_error": "prompt refused by policy"}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    lines = _stream_lines(gateway)
    events = [json.loads(line[6:]) for line in lines
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert events[0]["error"] == "prompt refused by policy"


def test_stream_over_a_json_only_engine_ignores_an_empty_error_string(monkeypatch, http_server):
    """{"error": "", "client_error": true} must not yield an empty error: an empty
    error string is no message at all, so client_error still applies (wAQ on #131)."""
    engine = http_server(json_engine({"error": "", "client_error": True}))
    gateway = http_server(Gateway)
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    lines = _stream_lines(gateway)
    events = [json.loads(line[6:]) for line in lines
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert events[0]["error"] == "%s: client error" % base_url(engine)
