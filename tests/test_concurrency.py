"""DH-P7-03: queued requests run one at a time in arrival order, the k>1 gate
switches off when the measured rate misses the floor, and GET /status carries
the running count. Floor and receipt: receipts/2026-11-12-concurrent-serving."""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster.gateway import Gateway, Slots


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


def run_clients(gateway, prompts, stagger=0.0):
    """One client thread per prompt. `stagger` seconds between starts pin the
    arrival (park) order at the gateway: a request parks ~2 ms after its start,
    far inside one stagger step."""
    results = []
    threads = []
    for i, prompt in enumerate(prompts):
        if i and stagger:
            time.sleep(stagger)
        t = threading.Thread(target=post_completion, args=(gateway, results, prompt))
        t.start()
        threads.append(t)
    return threads, results


def serve_slots(monkeypatch, k):
    monkeypatch.setattr(Gateway, "max_concurrent", k, raising=False)
    monkeypatch.setattr(Gateway, "slots", Slots(k))


def test_the_slot_gate_resizes_while_requests_hold_it():
    slots = Slots(2)
    assert slots.acquire(timeout=1)
    assert slots.acquire(timeout=1)
    slots.resize(1)  # the downshift while two hold
    assert not slots.acquire(timeout=0.05)  # held 2 >= 1: blocked
    slots.release()
    assert not slots.acquire(timeout=0.05)  # held 1 >= 1: still blocked — the downshift binds future entries
    slots.release()
    assert slots.acquire(timeout=0.05)  # held 0 < 1: serving one at a time
    slots.release()
    slots.resize(2)
    assert slots.acquire(timeout=0.05) and slots.acquire(timeout=0.05)  # the raise admits again


def test_the_slot_gate_serves_parked_waiters_in_arrival_order_through_a_downshift():
    slots = Slots(4)
    for _ in range(4):
        assert slots.acquire(timeout=1)
    order = []

    def waiter(i):
        if slots.acquire(timeout=5):
            order.append(i)
            slots.release()

    waiters = []
    for i in range(4):
        t = threading.Thread(target=waiter, args=(i,))
        t.start()
        waiters.append(t)
        time.sleep(0.05)  # one park step: each waiter is parked well before the next starts
    while len(slots._tickets) < 4:
        time.sleep(0.02)  # all four parked, in order
    slots.resize(1)
    for _ in range(4):
        slots.release()  # the four holders finish under the new limit
    for t in waiters:
        t.join(timeout=10)
    assert order == [0, 1, 2, 3]  # a downshift must not scramble the queue


def test_the_aggregate_rate_counts_clustered_completions_over_their_own_span():
    assert Gateway._aggregate_rate([(0.0, 0.25, 100), (1.0, 1.25, 100)]) == 160.0
    # 200 tokens over the 1.25 s the window really spans; a first-finish anchor
    # would have read 800 and kept the gate on below the floor.
    assert Gateway._aggregate_rate([(0.0, 1.0, 96), (1.0, 2.0, 96)]) == 96.0  # k=1 steady
    assert Gateway._aggregate_rate([(0.0, 1.0, 96)]) is None  # one sample: no window yet


def test_three_queued_requests_run_one_at_a_time_in_arrival_order(fresh_gate, monkeypatch, http_server):
    gate = GateEngine()
    engine = http_server(gate.handler())
    monkeypatch.setattr(Gateway, "engine", base_url(engine))
    serve_slots(monkeypatch, 1)
    gateway = http_server(Gateway)

    threads, results = run_clients(gateway, ["one"])
    assert gate.entered.acquire(timeout=5)
    queued, queued_results = run_clients(gateway, ["two", "three"], stagger=0.2)
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
    assert fresh_gate.slots.limit == 1  # both completions missed the floor: the gate switched off

    with gate.lock:
        gate.peak = 0
    late, late_results = run_clients(gateway, ["three", "four", "five"], stagger=0.2)
    assert gate.entered.acquire(timeout=5)
    assert not gate.entered.acquire(timeout=0.3)  # the other two queue one at a time
    for name in ("three", "four", "five"):
        gate._event(gate.gates, name).set()
        assert gate._event(gate.done, name).wait(5)
    for t in late:
        t.join(timeout=10)
    assert late_results == [200, 200, 200]
    assert sorted(gate.order) == ["five", "four", "one", "three", "two"]  # nothing lost
    assert gate.order[2:] == ["three", "four", "five"]  # the k=1 queue serves in arrival order
    assert gate.finished[2:] == gate.order[2:]
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
    assert fresh_gate.slots.limit == 2


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
