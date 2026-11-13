"""DH-P7-03: queued requests run one at a time in arrival order, the k>1 gate
switches off when the measured rate misses the floor, and GET /status carries
the running count. Floor and receipt: receipts/2026-11-12-concurrent-serving."""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from collections import deque
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


@pytest.fixture
def fresh_gate(monkeypatch):
    """Isolate the gateway's class-level gate state between tests."""
    monkeypatch.setattr(Gateway, "_measured", deque(maxlen=8), raising=False)
    monkeypatch.setattr(Gateway, "_running", 0, raising=False)
    return Gateway


class GateEngine:
    """Engine stub that holds every /generate until its own gate opens, records
    the order prompts enter and finish, and answers /status."""

    def __init__(self, status=None):
        self.lock = threading.Lock()
        self.inside = 0
        self.peak = 0
        self.order = []
        self.finished = []
        self.status = status or {"engine": "llama.cpp", "model": "m"}
        self.entered = threading.Semaphore(0)
        self.gates = {}
        self.done = {}

    def _event(self, table, prompt):
        with self.lock:
            return table.setdefault(prompt, threading.Event())

    def handler(self):
        gate = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _json(self, code, body):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/status":
                    return self._json(200, json.dumps(gate.status).encode())
                self._json(404, b'{"error": "not found"}')

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                prompt = json.loads(self.rfile.read(n) or "{}").get("prompt")
                with gate.lock:
                    gate.order.append(prompt)
                    gate.inside += 1
                    gate.peak = max(gate.peak, gate.inside)
                gate.entered.release()
                gate._event(gate.gates, prompt).wait(10)
                with gate.lock:
                    gate.inside -= 1
                    gate.finished.append(prompt)
                gate._event(gate.done, prompt).set()
                body = json.dumps({"text": "ok " + str(prompt), "prompt_tokens": 1,
                                   "completion_tokens": 96, "prefill_ms": 0.0,
                                   "decode_ms": 0.0, "tokps": 187.5}).encode()
                self._json(200, body)

        return Handler


def base_url(server):
    return f"http://127.0.0.1:{server.server_port}"


def post_completion(gateway, results, prompt):
    request = urllib.request.Request(
        base_url(gateway) + "/v1/completions",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            results.append(response.code)
    except urllib.error.HTTPError as e:
        results.append(e.code)
        e.close()


def run_clients(gateway, prompts):
    results = []
    threads = [threading.Thread(target=post_completion, args=(gateway, results, p))
               for p in prompts]
    for t in threads:
        t.start()
    return threads, results


def serve_slots(monkeypatch, k):
    monkeypatch.setattr(Gateway, "max_concurrent", k, raising=False)
    monkeypatch.setattr(Gateway, "slot_count", k, raising=False)
    monkeypatch.setattr(Gateway, "slots", threading.BoundedSemaphore(k))


def test_three_queued_requests_run_one_at_a_time_in_arrival_order(fresh_gate, monkeypatch, http_server):
    gate = GateEngine()
    engine = http_server(gate.handler())
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    serve_slots(monkeypatch, 1)
    gateway = http_server(Gateway)

    threads, results = run_clients(gateway, ["one"])
    assert gate.entered.acquire(timeout=5)
    queued, queued_results = run_clients(gateway, ["two", "three"])
    assert not gate.entered.acquire(timeout=0.3)  # two and three wait at the gateway

    for name in ("one", "two", "three"):
        gate._event(gate.gates, name).set()
        assert gate._event(gate.done, name).wait(5)
    for t in threads + queued:
        t.join(timeout=10)
    assert results + queued_results == [200, 200, 200]
    assert gate.order == ["one", "two", "three"]
    assert gate.finished == gate.order
    assert gate.peak == 1


def test_the_k2_gate_switches_off_when_the_fixture_floor_is_missed(fresh_gate, monkeypatch, http_server):
    gate = GateEngine()
    engine = http_server(gate.handler())
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    monkeypatch.setattr(Gateway, "floor_tokps", float("inf"))  # a floor no measured rate can meet
    serve_slots(monkeypatch, 2)
    gateway = http_server(Gateway)

    threads, results = run_clients(gateway, ["one", "two"])
    assert gate.entered.acquire(timeout=5)
    assert gate.entered.acquire(timeout=5)  # no rate yet: the configured k=2 runs
    assert not gate.entered.acquire(timeout=0.3)
    gate._event(gate.gates, "one").set()
    gate._event(gate.gates, "two").set()
    assert gate._event(gate.done, "one").wait(5)
    assert gate._event(gate.done, "two").wait(5)
    for t in threads:
        t.join(timeout=10)
    assert results == [200, 200]
    assert fresh_gate.slot_count == 1  # both completions missed the floor: the gate switched off

    with gate.lock:
        gate.peak = 0
    late, late_results = run_clients(gateway, ["three", "four", "five"])
    assert gate.entered.acquire(timeout=5)
    assert not gate.entered.acquire(timeout=0.3)  # the other two queue one at a time
    for name in ("three", "four", "five"):
        gate._event(gate.gates, name).set()
        assert gate._event(gate.done, name).wait(5)
    for t in late:
        t.join(timeout=10)
    assert late_results == [200, 200, 200]
    assert gate.order == ["one", "two", "three", "four", "five"]
    assert gate.finished == gate.order
    assert gate.peak == 1


def test_the_k2_gate_stays_on_above_the_floor(fresh_gate, monkeypatch, http_server):
    gate = GateEngine()
    engine = http_server(gate.handler())
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    monkeypatch.setattr(Gateway, "floor_tokps", 1.0)  # every measured rate clears it
    serve_slots(monkeypatch, 2)
    gateway = http_server(Gateway)

    threads, results = run_clients(gateway, ["one", "two"])
    assert gate.entered.acquire(timeout=5)
    assert gate.entered.acquire(timeout=5)
    for name in ("one", "two"):
        gate._event(gate.gates, name).set()
        assert gate._event(gate.done, name).wait(5)
    for t in threads:
        t.join(timeout=10)
    assert results == [200, 200]

    late, late_results = run_clients(gateway, ["three", "four"])
    assert gate.entered.acquire(timeout=5)
    assert gate.entered.acquire(timeout=5)  # the measured rate is above the floor: k=2 still runs
    assert not gate.entered.acquire(timeout=0.3)
    for name in ("three", "four"):
        gate._event(gate.gates, name).set()
        assert gate._event(gate.done, name).wait(5)
    for t in late:
        t.join(timeout=10)
    assert late_results == [200, 200]
    assert gate.peak == 2
    assert fresh_gate.slot_count == 2


def test_status_reports_the_running_request_count(fresh_gate, monkeypatch, http_server):
    gate = GateEngine(status={"engine": "llama.cpp", "model": "m"})
    engine = http_server(gate.handler())
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    serve_slots(monkeypatch, 2)
    gateway = http_server(Gateway)

    threads, results = run_clients(gateway, ["one", "two"])
    assert gate.entered.acquire(timeout=5)
    assert gate.entered.acquire(timeout=5)
    with urllib.request.urlopen(base_url(gateway) + "/status", timeout=2) as response:
        status = json.load(response)
    assert status["running"] == 2
    assert status["engine"] == "llama.cpp"  # the engine's own status is still merged in

    for name in ("one", "two"):
        gate._event(gate.gates, name).set()
        assert gate._event(gate.done, name).wait(5)
    for t in threads:
        t.join(timeout=10)
    assert results == [200, 200]
    with urllib.request.urlopen(base_url(gateway) + "/status", timeout=2) as response:
        assert json.load(response)["running"] == 0
