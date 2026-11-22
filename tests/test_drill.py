"""Node-loss drill: a real HTTP worker, a real mid-request socket reset, a real
restart — all in-process on localhost, so no cluster hardware and no spawned
processes are needed (CI runners may refuse listeners in spawned children)."""
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import agent, drill

SLOW_S = 3.0  # request work time inside the stub, long enough to kill mid-flight


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _tcp_up(port, timeout=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.3).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


class Worker:
    """The stand-in for the RPC worker: an HTTP server that answers after a
    delay. kill() stops the listener and RSTs every in-flight connection —
    on the wire that is exactly what a SIGKILLed process looks like."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            w = self.server.worker
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            w.connections.append(self.connection)
            time.sleep(w.delay)
            body = json.dumps({"reply": "stable-answer"}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # the client side was already RST by kill()

    def __init__(self, delay=SLOW_S):
        self.delay = delay
        self.connections = []
        self.server = None
        self.port = None

    def start(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Worker.Handler)
        srv.daemon_threads = True
        srv.worker = self
        self.server = srv
        self.port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    def kill(self):
        self.server.shutdown()
        self.server.server_close()
        for conn in self.connections:
            try:
                # socket.close() defers to the rfile/wfile io refs and leaves
                # the fd open, so detach the fd and close it with LINGER=0:
                # that sends RST — the wire signature of a killed process.
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                struct.pack("ii", 1, 0))
                os.close(conn.detach())
            except OSError:
                pass

    def wait_ready(self, _remaining_s):
        return _tcp_up(self.port)

    def request(self):
        req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % self.port,
                                     data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())


@pytest.fixture
def worker():
    w = Worker()
    w.start()
    yield w
    try:
        w.server.shutdown()
        w.server.server_close()
    except (AttributeError, OSError):
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


def test_rpc_stop_signals_every_listener(monkeypatch):
    """A real socket LISTENing in this process: listener_pids must find this
    pid, _signal_all must get SIGKILL, and the port must go quiet afterwards."""
    lsock = socket.socket()
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen()
    port = lsock.getsockname()[1]
    signaled = []

    def fake_signal_all(pids, sig):
        signaled.append((sorted(pids), sig))
        lsock.close()  # the "death": the listener goes away, as after SIGKILL

    monkeypatch.setattr(agent, "_signal_all", fake_signal_all)
    res = agent.rpc_stop({"port": port})
    assert signaled == [([os.getpid()], signal.SIGKILL)]
    assert res == {"killed": [os.getpid()], "still_listening": []}
    assert not _tcp_up(port)


def test_rpc_stop_no_listener_is_a_noop():
    res = agent.rpc_stop({"port": _free_port()})
    assert res == {"killed": [], "still_listening": []}


def test_rpc_stop_validates_port():
    with pytest.raises(ValueError):
        agent.rpc_stop({"port": "not-a-port"})
