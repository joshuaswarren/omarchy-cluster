"""Node-loss drill: a real HTTP worker, a real mid-request socket reset, a real
restart — all in-process on localhost, so no cluster hardware and no spawned
processes are needed (CI runners may refuse listeners in spawned children)."""
import io
import json
import os
import shlex
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

from omarchy_cluster import agent, cli, client, drill

SLOW_S = 1.5  # request work time inside the stub, long enough to kill mid-flight


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
            body = json.dumps(w.make_reply()).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # the client side was already RST by kill()

    def __init__(self, delay=SLOW_S, reply_fn=None, fixed_port=0):
        self.delay = delay
        self._reply_fn = reply_fn or (lambda: {"reply": "stable-answer"})
        self._fixed_port = fixed_port
        self.connections = []
        self.server = None
        self.port = None

    def make_reply(self):
        return self._reply_fn()

    def start(self, port=None):
        bind_port = self._fixed_port if port is None else port
        srv = ThreadingHTTPServer(("127.0.0.1", bind_port), Worker.Handler)
        srv.allow_reuse_address = True
        srv.daemon_threads = True
        srv.worker = self
        self.server = srv
        self.port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    def kill(self):
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        self.server = None
        for conn in list(self.connections):
            try:
                # socket.close() defers to the rfile/wfile io refs and leaves
                # the fd open, so detach the fd and close it with LINGER=0:
                # that sends RST — the wire signature of a killed process.
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                struct.pack("ii", 1, 0))
                os.close(conn.detach())
            except OSError:
                pass
        self.connections.clear()

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
        if w.server is not None:
            w.server.shutdown()
            w.server.server_close()
    except (AttributeError, OSError):
        pass


def test_drill_passes_on_kill_and_restart(worker):
    receipt = drill.run_drill(worker.request, worker.kill, restart=worker.start,
                              wait_ready=worker.wait_ready,
                              kill_after_s=0.25, error_deadline_s=10.0, ready_deadline_s=15.0)
    assert receipt["verdict"] == "pass", receipt
    # the in-flight request ended with a bounded error, not a bogus 200
    inflight = receipt["steps"]["inflight_kill"]
    assert inflight["ok"] is False
    assert inflight["after_kill_s"] < 10.0
    # the replay got the same answer the baseline did
    assert receipt["steps"]["replay"]["body"] == receipt["steps"]["baseline"]["body"]


def test_drill_fails_when_request_survives_the_kill(worker):
    # a kill that misses leaves the request to finish — that is not node loss
    receipt = drill.run_drill(worker.request, lambda: None,
                              kill_after_s=0.25, error_deadline_s=10.0)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "inflight_kill"


def test_drill_fails_when_request_hangs_on_loss():
    calls = {"n": 0}
    hang = threading.Event()

    def request():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "ok"}
        hang.wait(10)
        return {"reply": "late"}

    try:
        receipt = drill.run_drill(request, lambda: None,
                                  kill_after_s=0.05, error_deadline_s=0.3)
    finally:
        hang.set()
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


def test_drill_passes_with_openai_gateway_response_shape():
    """Real omarchy-cluster gateway replies carry per-request id, created, and
    timings; replay must pass when generated choices and usage match."""
    calls = {"n": 0}

    def gateway_request():
        calls["n"] += 1
        if calls["n"] in (2, 3):
            raise urllib.error.HTTPError(
                "http://127.0.0.1:8020/v1/completions", 502, "Bad Gateway", {}, None)
        return {
            "id": "chatcmpl-omarchy-%d" % (1700000000 + calls["n"]),
            "created": 1700000000 + calls["n"],
            "model": "omarchy-cluster",
            "object": "text_completion",
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            "timings": {"prefill_ms": 100.0 + calls["n"], "decode_ms": 200.0 + calls["n"],
                        "tokens_per_s": 9.5 + calls["n"] * 0.1},
            "choices": [{"index": 0, "text": "Loss teaches resilience.", "finish_reason": "stop"}],
        }

    receipt = drill.run_drill(gateway_request, lambda: None, restart=lambda: None,
                              wait_ready=lambda _s: True, kill_after_s=0.0)
    assert receipt["verdict"] == "pass", receipt


def test_drill_fails_when_openai_gateway_choices_differ():
    calls = {"n": 0}

    def gateway_request():
        calls["n"] += 1
        if calls["n"] in (2, 3):
            raise urllib.error.URLError("connection reset")
        text = "Loss teaches resilience." if calls["n"] == 1 else "Corrupted tokens."
        return {
            "id": "chatcmpl-omarchy-%d" % calls["n"],
            "created": 1700000000 + calls["n"],
            "model": "omarchy-cluster",
            "object": "text_completion",
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        }

    receipt = drill.run_drill(gateway_request, lambda: None, restart=lambda: None,
                              wait_ready=lambda _s: True, kill_after_s=0.0)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "replay"
    assert "different output" in receipt["reason"]


def test_drill_inflight_deadline_counts_post_kill_time():
    """Pre-kill run time (kill_after_s) must not eat into error_deadline_s."""
    killed = threading.Event()
    calls = {"n": 0}

    def request():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "ok"}
        if calls["n"] == 2:
            killed.wait(5)
            time.sleep(0.05)
            raise urllib.error.URLError("connection reset")
        if calls["n"] == 3:
            raise urllib.error.URLError("connection refused")
        return {"reply": "ok"}

    receipt = drill.run_drill(request, killed.set, restart=lambda: None,
                              wait_ready=lambda _s: True,
                              kill_after_s=0.5, error_deadline_s=0.4)
    assert receipt["verdict"] == "pass", receipt
    assert receipt["steps"]["inflight_kill"]["elapsed_s"] >= 0.5
    assert receipt["steps"]["inflight_kill"]["after_kill_s"] < 0.4


def test_drill_fails_when_dead_worker_succeeds():
    calls = {"n": 0}

    def request():
        calls["n"] += 1
        if calls["n"] == 2:
            raise urllib.error.URLError("connection reset")
        return {"reply": "ok"}  # step 3 unexpectedly succeeds

    receipt = drill.run_drill(request, lambda: None, kill_after_s=0.0)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "dead_worker"


def test_drill_fails_when_dead_worker_hangs():
    calls = {"n": 0}
    hang = threading.Event()

    def request():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "ok"}
        if calls["n"] == 2:
            raise urllib.error.URLError("connection reset")
        hang.wait(10)
        return {"reply": "late"}

    try:
        receipt = drill.run_drill(request, lambda: None,
                                  kill_after_s=0.0, error_deadline_s=0.25)
    finally:
        hang.set()
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "dead_worker"


def test_drill_fails_cleanly_when_kill_raises():
    def bad_kill():
        raise RuntimeError("agent unreachable")

    receipt = drill.run_drill(lambda: {"reply": "ok"}, bad_kill, kill_after_s=0.0)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "inflight_kill"
    assert "kill failed" in receipt["reason"]


def test_drill_fails_cleanly_when_kill_leaves_listener():
    receipt = drill.run_drill(
        lambda: {"reply": "ok"},
        lambda: {"killed": [123], "still_listening": [123]},
        kill_after_s=0.0)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "inflight_kill"
    assert "still listening" in receipt["reason"]


def test_drill_fails_cleanly_when_restart_raises():
    calls = {"n": 0}

    def request():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "ok"}
        raise urllib.error.URLError("connection refused")

    def bad_restart():
        raise RuntimeError("rpc_start failed")

    receipt = drill.run_drill(request, lambda: None, restart=bad_restart, kill_after_s=0.0)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "restart"
    assert "restart failed" in receipt["reason"]


def test_drill_no_restart_skips_recovery():
    calls = {"n": 0}

    def request():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "ok"}
        raise urllib.error.URLError("connection refused")

    receipt = drill.run_drill(request, lambda: None, restart=None, kill_after_s=0.0)
    assert receipt["verdict"] == "pass"
    assert "skipped" in receipt["recovery"]


def test_drill_fails_when_restart_never_readies():
    calls = {"n": 0}

    def dead():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "ok"}
        raise urllib.error.URLError("connection refused")

    receipt = drill.run_drill(dead, lambda: None, restart=lambda: None,
                              wait_ready=lambda _s: False, kill_after_s=0.0,
                              ready_deadline_s=0.4)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "restart"


def test_drill_gateway_restart_and_health_recovery():
    events = []
    calls = {"n": 0}

    def request():
        calls["n"] += 1
        if calls["n"] in (2, 3):
            raise urllib.error.URLError("connection reset")
        return {"reply": "ok"}

    receipt = drill.run_drill(
        request,
        kill=lambda: events.append("kill"),
        restart=lambda: events.append("restart_worker"),
        wait_ready=lambda _s: (events.append("wait_worker") or True),
        restart_gateway=lambda: events.append("restart_gateway"),
        wait_gateway_ready=lambda _s: (events.append("wait_gateway") or True),
        kill_after_s=0.0)
    assert receipt["verdict"] == "pass", receipt
    assert events == ["kill", "restart_worker", "wait_worker", "restart_gateway", "wait_gateway"]
    assert receipt["steps"]["restart_gateway"]["ready"] is True


def test_drill_fails_when_gateway_never_readies():
    calls = {"n": 0}

    def request():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "ok"}
        raise urllib.error.URLError("connection reset")

    receipt = drill.run_drill(
        request, lambda: None, restart=lambda: None,
        wait_ready=lambda _s: True,
        restart_gateway=lambda: {"restarted": True},
        wait_gateway_ready=lambda _s: False,
        kill_after_s=0.0, ready_deadline_s=0.3)
    assert receipt["verdict"] == "fail"
    assert receipt["step"] == "restart_gateway"


@pytest.mark.parametrize("raw,expected", [
    ("10.0.0.2", ("10.0.0.2", 8025)),
    ("10.0.0.2:8025", ("10.0.0.2", 8025)),
    ("10.0.0.2:9000", ("10.0.0.2", 9000)),
    ("mac-studio.local", ("mac-studio.local", 8025)),
    ("mac-studio.local:8443", ("mac-studio.local", 8443)),
    ("[::1]", ("::1", 8025)),
    ("[::1]:9000", ("::1", 9000)),
    ("::1", ("::1", 8025)),
])
def test_parse_node_address(raw, expected):
    assert drill.parse_node_address(raw) == expected


@pytest.mark.parametrize("bad", [
    "", "   ", ":8025", "10.0.0.2:", "10.0.0.2:0", "10.0.0.2:70000",
    "10.0.0.2:notaport", "[::1", "[]", "[::1]extra", "not:an:ipv6",
])
def test_parse_node_address_rejects_invalid(bad):
    with pytest.raises(ValueError):
        drill.parse_node_address(bad)


def _start_stub_gateway(rpc_worker, expected_token, marker_path=None):
    """In-process gateway stand-in serving /health and /v1/completions with
    OpenAI-shaped responses, forwarding inference to rpc_worker."""
    state = {"seq": 0, "needs_gw_restart": False, "auth_headers": [], "bodies": []}

    class GatewayHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _healthy(self):
            if rpc_worker.server is None:
                return False
            if state["needs_gw_restart"]:
                if marker_path and os.path.exists(marker_path):
                    state["needs_gw_restart"] = False
                else:
                    return False
            return True

        def do_GET(self):
            if self.path == "/health":
                if self._healthy():
                    return self._send_json(200, {"ok": True})
                return self._send_json(503, {"ok": False})
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/completions":
                return self._send_json(404, {"error": "not found"})
            auth = self.headers.get("Authorization")
            state["auth_headers"].append(auth)
            if auth != "Bearer " + expected_token:
                return self._send_json(401, {"error": "unauthorized"})
            n = int(self.headers.get("Content-Length") or 0)
            req_body = json.loads(self.rfile.read(n) or b"{}")
            state["bodies"].append(req_body)
            if not self._healthy():
                return self._send_json(502, {"error": "rpc worker down"})
            try:
                worker_reply = rpc_worker.request()
            except OSError as e:
                state["needs_gw_restart"] = marker_path is not None
                return self._send_json(502, {"error": "rpc worker lost: %s" % e})
            state["seq"] += 1
            return self._send_json(200, {
                "id": "chatcmpl-omarchy-%d" % (1700000000 + state["seq"]),
                "created": 1700000000 + state["seq"],
                "model": req_body.get("model") or "omarchy-cluster",
                "object": "text_completion",
                "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
                "timings": {"prefill_ms": 10.0 + state["seq"], "decode_ms": 20.0 + state["seq"],
                            "tokens_per_s": 50.0 + state["seq"]},
                "choices": [{"index": 0, "text": worker_reply["reply"], "finish_reason": "stop"}],
            })

    srv = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def test_cmd_drill_end_to_end_without_hardware(tmp_path, monkeypatch, capsys):
    """End-to-end CLI test: --node without a port (default 8025), real mid-request
    worker RST, --rpc-body, --restart-gateway-cmd, OpenAI gateway reply with
    changing id/created/timings, Bearer auth, and --out receipt."""
    tok = "secret-cluster-token"
    tok_file = tmp_path / "token"
    tok_file.write_text(tok + "\n")
    rpc_port = _free_port()
    rpc_body_file = tmp_path / "rpc-body.json"
    rpc_body_file.write_text(json.dumps({"device": "Vulkan0"}))
    marker = tmp_path / "gw-restarted"
    out_file = tmp_path / "receipts" / "drill.json"

    w = Worker(delay=SLOW_S, fixed_port=rpc_port)
    w.start()
    gw, gw_state = _start_stub_gateway(w, tok, marker_path=str(marker))
    agent_calls = []

    def fake_agent_urlopen(req, timeout):
        assert req.headers.get("X-cluster-token") == tok
        payload = json.loads(req.data.decode())
        agent_calls.append((req.full_url, payload))
        if req.full_url.endswith("/v1/rpc/stop"):
            w.kill()
            resp = {"killed": [99999], "still_listening": []}
        elif req.full_url.endswith("/v1/rpc/start"):
            w.start(port=payload["port"])
            resp = {"pid": 99998, "cmd": ["ggml-rpc-server"]}
        else:
            raise AssertionError("unexpected agent call: %s" % req.full_url)
        return io.BytesIO(json.dumps(resp).encode())

    monkeypatch.setattr(client, "urlopen", fake_agent_urlopen)
    restart_cmd = "%s -c %s" % (
        shlex.quote(sys.executable),
        shlex.quote("open(%r, 'w').write('ok')" % str(marker)),
    )
    try:
        cli.main([
            "drill",
            "--node", "127.0.0.1",  # no :port — must default to 8025
            "--rpc-port", str(rpc_port),
            "--gateway", "http://127.0.0.1:%d" % gw.server_port,
            "--model", "qwen3-1.7b",
            "--kill-after", "0.25",
            "--error-deadline", "10",
            "--ready-deadline", "10",
            "--rpc-body", str(rpc_body_file),
            "--restart-gateway-cmd", restart_cmd,
            "--token-file", str(tok_file),
            "--out", str(out_file),
        ])
    finally:
        gw.shutdown()
        gw.server_close()
        w.kill()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(out_file.read_text())
    assert printed == written
    assert printed["verdict"] == "pass", printed
    assert printed["steps"]["restart_gateway"]["ready"] is True
    assert agent_calls == [
        ("https://127.0.0.1:8025/v1/rpc/stop", {"port": rpc_port}),
        ("https://127.0.0.1:8025/v1/rpc/start", {"device": "Vulkan0", "port": rpc_port}),
    ]
    assert all(h == "Bearer " + tok for h in gw_state["auth_headers"])
    assert gw_state["bodies"][0]["temperature"] == 0
    assert gw_state["bodies"][0]["model"] == "qwen3-1.7b"


def test_cmd_drill_no_restart_and_failure_exit_code(tmp_path, monkeypatch, capsys):
    tok = "secret-cluster-token"
    tok_file = tmp_path / "token"
    tok_file.write_text(tok)
    rpc_port = _free_port()
    w = Worker(delay=SLOW_S, fixed_port=rpc_port)
    w.start()
    gw, _ = _start_stub_gateway(w, tok)
    agent_calls = []

    def fake_agent_urlopen(req, timeout):
        agent_calls.append(req.full_url)
        if req.full_url.endswith("/v1/rpc/stop"):
            w.kill()
            return io.BytesIO(json.dumps({"killed": [1], "still_listening": []}).encode())
        raise AssertionError("should not restart when --no-restart is set")

    monkeypatch.setattr(client, "urlopen", fake_agent_urlopen)
    try:
        rc = drill.main([
            "--node", "127.0.0.1:9443",
            "--rpc-port", str(rpc_port),
            "--gateway", "http://127.0.0.1:%d" % gw.server_port,
            "--kill-after", "0.25",
            "--error-deadline", "10",
            "--no-restart",
            "--token-file", str(tok_file),
        ])
        assert rc == 0
        receipt = json.loads(capsys.readouterr().out)
        assert receipt["verdict"] == "pass"
        assert "skipped" in receipt["recovery"]
        assert agent_calls == ["https://127.0.0.1:9443/v1/rpc/stop"]

        # Now worker is already dead: baseline fails, exit code is 1, and --out is still written.
        fail_out = tmp_path / "fail.json"
        rc_fail = drill.main([
            "--node", "127.0.0.1:9443",
            "--rpc-port", str(rpc_port),
            "--gateway", "http://127.0.0.1:%d" % gw.server_port,
            "--token-file", str(tok_file),
            "--out", str(fail_out),
        ])
        assert rc_fail == 1
        fail_receipt = json.loads(fail_out.read_text())
        assert fail_receipt["verdict"] == "fail"
        assert fail_receipt["step"] == "baseline"
    finally:
        gw.shutdown()
        gw.server_close()
        w.kill()


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
