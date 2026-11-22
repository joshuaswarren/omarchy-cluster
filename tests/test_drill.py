"""Node-loss drill: a real worker process, a real SIGKILL mid-request, a real
restart — all on localhost, so no cluster hardware is needed."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import agent, drill

# The stand-in for the RPC worker: an HTTP server that answers after a delay.
# SIGKILL mid-request drops the socket, exactly like a crashed ggml-rpc-server.
STUB = """
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        time.sleep(float(sys.argv[2]))
        body = json.dumps({"reply": "stable-answer"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""

SLOW_S = 4.0  # request work time inside the stub, long enough to kill mid-flight


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _tcp_up(port, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.3).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


class StubWorker:
    """A real process listening on a real port; kill() is SIGKILL."""

    def __init__(self, delay=SLOW_S):
        self.port = _free_port()
        self.delay = delay
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-c", STUB, str(self.port), str(self.delay)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        assert _tcp_up(self.port), "stub worker did not listen on %d" % self.port

    def kill(self):
        self.proc.kill()
        self.proc.wait(5)

    def wait_ready(self, _remaining_s):
        return _tcp_up(self.port, timeout=0.5)

    def request(self):
        req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % self.port,
                                     data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())


@pytest.fixture
def worker():
    w = StubWorker()
    w.start()
    try:
        yield w
    finally:
        try:
            w.proc.kill()
            w.proc.wait(5)
        except (ProcessLookupError, OSError):
            pass


def test_drill_passes_on_kill_and_restart(worker):
    receipt = drill.run_drill(worker.request, worker.kill, restart=worker.start,
                            wait_ready=worker.wait_ready,
                            kill_after_s=0.3, error_deadline_s=10.0, ready_deadline_s=15.0)
    assert receipt["verdict"] == "pass", receipt
    # the in-flight request ended with a bounded error, not a bogus 200
    inflight = receipt["steps"]["inflight_kill"]
    assert inflight["ok"] is False
    assert inflight["elapsed_s"] < 10.0
    # the replay got the same answer the baseline did
    assert receipt["steps"]["replay"]["body"] == receipt["steps"]["baseline"]["body"]


def test_drill_fails_when_request_survives_the_kill(worker):
    # a kill that misses leaves the request to finish — that is not node loss
    receipt = drill.run_drill(worker.request, lambda: None,
                            kill_after_s=0.3, error_deadline_s=10.0)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "inflight_kill"


def test_drill_fails_when_request_hangs_on_loss():
    hang = threading.Event()
    receipt = drill.run_drill(lambda: hang.wait(60), lambda: None,
                            kill_after_s=0.05, error_deadline_s=0.5)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "inflight_kill"
    assert "hang" in receipt["reason"]


def test_drill_fails_when_baseline_broken():
    receipt = drill.run_drill(lambda: 1 / 0, lambda: None)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "baseline"


def test_drill_fails_on_different_replay_output():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] in (2, 3):  # in-flight request and dead-worker probe both die
            raise urllib.error.URLError("connection reset")
        return {"reply": "answer-%d" % calls["n"]}  # baseline=1, replay=4: differ
    receipt = drill.run_drill(flaky, lambda: None, restart=lambda: None,
                            wait_ready=lambda _s: True, kill_after_s=0.0)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "replay"
    assert "different output" in receipt["reason"]


def test_drill_fails_when_restart_never_readies():
    calls = {"n": 0}
    def dead():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "ok"}
        raise urllib.error.URLError("connection refused")
    receipt = drill.run_drill(dead, lambda: None, restart=lambda: None,
                            wait_ready=lambda _s: False, kill_after_s=0.0,
                            ready_deadline_s=0.6)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "restart"


def test_rpc_stop_kills_the_listener():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-c", STUB, str(port), "60"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    try:
        assert _tcp_up(port)
        res = agent.rpc_stop({"port": port})
        assert proc.pid in res["killed"]
        assert res["still_listening"] == []
        proc.wait(5)
        assert proc.returncode == -9  # SIGKILL, not a graceful close
    finally:
        try:
            proc.kill()
            proc.wait(5)
        except (ProcessLookupError, OSError):
            pass


def test_rpc_stop_no_listener_is_a_noop():
    res = agent.rpc_stop({"port": _free_port()})
    assert res == {"killed": [], "still_listening": []}


def test_rpc_stop_validates_port():
    with pytest.raises(ValueError):
        agent.rpc_stop({"port": "not-a-port"})
