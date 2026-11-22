"""Joint-engine (prefill -> KV handoff -> decode) loopback/mock tests.

Two fake llama-servers stand in for Spark (prefill) and Mac (decode). The KV
file hops through each fake's slot-save dir the way --slot-save-path would, so
the whole phase chain runs on loopback without hardware.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import joint_engine as je
from omarchy_cluster import llamacpp_engine as lce


class FakeLlama:
    """A fake llama-server: /completion, /slots, /tokenize, /health.

    save writes `kv_bytes` into its slot dir under the requested filename;
    restore reads it back the same way the real endpoints do."""

    def __init__(self, slot_dir, text="decoded text", prompt_n=8192, predicted_n=64,
                 predicted_per_second=12.5, fail_completion=None, fail_slots=None):
        self.slot_dir = slot_dir
        self.text = text
        self.prompt_n = prompt_n
        self.predicted_n = predicted_n
        self.predicted_per_second = predicted_per_second
        self.fail_completion = fail_completion
        self.fail_slots = fail_slots or {}
        self.requests = []  # (path, parsed body) for assertions
        self.events = []  # ordered phase log: "completion", "save", "restore", "tokenize"


def _handler(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._reply(200, {"status": "ok"})
            return self._reply(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._reply(400, {"error": "bad json"})
            fake.requests.append((self.path, req))
            parsed = urlparse(self.path)
            if parsed.path == "/completion":
                if fake.fail_completion:
                    return self._reply(fake.fail_completion, {"error": "boom"})
                fake.events.append("completion")
                if req.get("n_predict") == 0:
                    return self._reply(400, {"error": "n_predict must be > 0"})
                return self._reply(200, {
                    "content": fake.text, "stop_type": "limit",
                    "timings": {"prompt_n": fake.prompt_n, "prompt_ms": 900.0,
                                "predicted_n": fake.predicted_n, "predicted_ms": 5000.0,
                                "predicted_per_second": fake.predicted_per_second}})
            if parsed.path.startswith("/slots/"):
                action = parse_qs(parsed.query).get("action", [""])[0]
                if fake.fail_slots.get(action):
                    return self._reply(fake.fail_slots[action], {"error": "slots disabled"})
                fake.events.append(action)
                fname = req.get("filename", "")
                path = os.path.join(fake.slot_dir, fname)
                if action == "save":
                    data = b"KV-BYTES-%d" % fake.prompt_n
                    with open(path, "wb") as f:
                        f.write(data)
                    return self._reply(200, {"id_slot": 0, "filename": fname,
                                             "n_saved": fake.prompt_n + 1, "n_written": len(data),
                                             "timings": {"save_ms": 40.0}})
                if action == "restore":
                    if not os.path.exists(path):
                        return self._reply(400, {"error": "cannot open file"})
                    fake.events.append("restored:%d" % os.path.getsize(path))
                    return self._reply(200, {"id_slot": 0, "filename": fname,
                                             "n_restored": fake.prompt_n + 1,
                                             "n_read": os.path.getsize(path),
                                             "timings": {"restore_ms": 40.0}})
                if action == "erase":
                    return self._reply(200, {"id_slot": 0, "n_erased": 10})
            if parsed.path == "/tokenize":
                fake.events.append("tokenize")
                return self._reply(200, {"tokens": [11, 22, 33]})
            return self._reply(404, {"error": "not found"})

    return H


@pytest.fixture
def llama_pair(tmp_path):
    """Two fake llama-servers with their own slot dirs. Yields (prefill, decode)."""
    prefill = FakeLlama(str(tmp_path / "pf"))
    decode = FakeLlama(str(tmp_path / "dec"))
    servers = []
    for fake in (prefill, decode):
        os.makedirs(fake.slot_dir)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        fake.url = "http://127.0.0.1:%d" % srv.server_port
        servers.append(srv)
    yield prefill, decode
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def cfg_for(prefill, decode, **kw):
    cfg = {"prefill_url": prefill.url, "decode_url": decode.url,
           "prefill_kv": je.LocalKv(prefill.slot_dir),
           "decode_kv": je.LocalKv(decode.slot_dir),
           "slot": 0, "kv_name": "joint-kv.bin",
           "probes": {}}
    cfg.update(kw)
    return cfg


def test_prefill_save_transport_restore_decode(llama_pair):
    prefill, decode = llama_pair
    res = je.joint_generate(cfg_for(prefill, decode), {"prompt": "p" * 8000, "max_tokens": 64})
    assert res["text"] == "decoded text"
    assert res["completion_tokens"] == 64
    # prefill was asked to warm the slot only (one token), decode ran the real request
    pf_req = [r for p, r in prefill.requests if p == "/completion"][0]
    assert pf_req["n_predict"] == 1 and pf_req["id_slot"] == 0
    dc_req = [r for p, r in decode.requests if p == "/completion"][0]
    assert dc_req["n_predict"] == 64 and dc_req["id_slot"] == 0
    assert dc_req["prompt"] == "p" * 8000  # same prompt; the cached prefix makes it free
    # the phases ran in order
    assert prefill.events[:2] == ["completion", "save"]
    assert "restore" in decode.events and decode.events.index("restore") < decode.events.index("completion")


def test_telemetry_eight_metrics(llama_pair):
    prefill, decode = llama_pair
    res = je.joint_generate(cfg_for(prefill, decode), {"prompt": "hello", "max_tokens": 8})
    tel = res["telemetry"]
    assert tel["token_ids_sha256"]  # from decode /tokenize
    assert tel["ttft_ms"] > 0
    assert tel["decode_tokps"] == 12.5
    assert tel["total_s"] > 0
    assert tel["kv_bytes"] == len(b"KV-BYTES-%d" % prefill.prompt_n)
    assert tel["errors"] == []
    assert tel["transport"] == "loopback"
    assert tel["lan_mbps"] is None
    assert "power_watts" in tel
    assert tel["phases"]["prefill_ms"] == 900.0
    assert tel["phases"]["restore_ms"] == 40.0


def test_prefill_error_propagates(llama_pair):
    prefill, decode = llama_pair
    prefill.fail_completion = 500
    res = je.joint_generate(cfg_for(prefill, decode), {"prompt": "x", "max_tokens": 4})
    assert "error" in res
    assert decode.requests == []  # never reached decode
    assert res["telemetry"]["errors"]


def test_save_endpoint_missing_tells_about_slot_save_path(llama_pair):
    prefill, decode = llama_pair
    prefill.fail_slots = {"save": 404}
    res = je.joint_generate(cfg_for(prefill, decode), {"prompt": "x", "max_tokens": 4})
    assert "error" in res
    assert "slot-save-path" in res["error"] or "slots" in res["error"]


def test_restore_missing_file(llama_pair):
    prefill, decode = llama_pair
    # make the transport write nowhere: decode kv dir without the file
    cfg = cfg_for(prefill, decode)
    res = je.joint_generate(cfg, {"prompt": "x", "max_tokens": 4}, transport=lambda n, s, d: {"bytes": 0, "seconds": 0.001, "path": "test"})
    assert "error" in res
    assert res["telemetry"]["errors"]


def test_remote_transport_counts_lan(llama_pair, monkeypatch):
    prefill, decode = llama_pair
    pulled = {}

    def fake_pull(ip, port, name, dest, timeout=600):
        data = open(os.path.join(prefill.slot_dir, name), "rb").read()
        with open(dest, "wb") as f:
            f.write(data)
        pulled["n"] = len(data)
        return len(data)

    monkeypatch.setattr(je, "_agent_pull", fake_pull)
    cfg = cfg_for(prefill, decode, prefill_kv=je.AgentKv("10.0.0.9", 8025))
    res = je.joint_generate(cfg, {"prompt": "x", "max_tokens": 4})
    assert "error" not in res
    assert pulled["n"] == len(b"KV-BYTES-%d" % prefill.prompt_n)
    assert res["telemetry"]["transport"] == "lan"
    assert res["telemetry"]["lan_mbps"] > 0


def test_requests_serialized(llama_pair):
    """Two concurrent joint requests must not interleave on the shared slot file."""
    prefill, decode = llama_pair
    order = []
    lock = threading.Lock()
    orig = je.transport_kv

    def watched(name, src, dst):
        with lock:
            order.append("t0")
        time.sleep(0.02)
        r = orig(name, src, dst)
        with lock:
            order.append("t1")
        return r

    cfg = cfg_for(prefill, decode)
    errs = []

    def one():
        res = je.joint_generate(cfg, {"prompt": "x", "max_tokens": 4}, transport=watched)
        if "error" in res:
            errs.append(res["error"])

    threads = [threading.Thread(target=one) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs
    assert order == ["t0", "t1", "t0", "t1"]  # each transport completed before the next began


def test_sampling_reaches_both_servers(llama_pair):
    prefill, decode = llama_pair
    res = je.joint_generate(cfg_for(prefill, decode),
                            {"prompt": "x", "max_tokens": 4,
                             "sampling": {"temperature": 0.4, "seed": 7}})
    assert "error" not in res
    for fake in (prefill, decode):
        req = [r for p, r in fake.requests if p == "/completion"][0]
        assert req.get("seed") == 7 and req.get("temperature") == 0.4


def test_health_checks_both(llama_pair):
    prefill, decode = llama_pair
    assert je.healthy(cfg_for(prefill, decode)) is True
    decode.url = "http://127.0.0.1:1"  # dead
    assert je.healthy(cfg_for(prefill, decode)) is False


def test_serve_joint_spawns_decode_then_engine(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from omarchy_cluster import cli
    kv_dir = str(tmp_path / "kv")
    spawned = []
    monkeypatch.setattr(lce, "gguf_info", lambda p: {"n_layer": 4, "total_bytes": 4 * 10 ** 9})
    monkeypatch.setattr(lce, "wait_http", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_write_state", lambda s: None)
    monkeypatch.setattr(cli, "_start_gateway", lambda *a: spawned.append(("gateway", a[2])))
    monkeypatch.setattr(cli, "_spawn",
                        lambda cmd, log_name, env=None: spawned.append((log_name, cmd))
                        or (SimpleNamespace(pid=1), log_name))

    class _Props:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps({"model_path": "/x/model.gguf"}).encode()
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: _Props())
    import omarchy_cluster.facts as facts_mod
    monkeypatch.setattr(facts_mod, "perf_cores", lambda: 4)

    args = SimpleNamespace(model=str(tmp_path / "model.gguf"), engine="joint",
                           prefill_url="http://prefill:8090", prefill_kv="10.0.0.9:8025",
                           kv_dir=kv_dir, joint_slot=0, joint_kv_name="joint-kv.bin",
                           llama_server="llama-server", llama_arg=[], ctx=4096,
                           engine_port=8031, port=8020, max_concurrent=4)
    cli._serve_joint(args)
    kinds = [s[0] for s in spawned]
    assert kinds == ["joint-decode-llama.log", "joint-engine.log", "gateway"]
    decode_cmd = spawned[0][1]
    assert decode_cmd[0] == "llama-server" and "--slot-save-path" in decode_cmd
    assert decode_cmd[decode_cmd.index("--slot-save-path") + 1] == kv_dir
    assert "--port" in decode_cmd and decode_cmd[decode_cmd.index("--port") + 1] == "8032"
    joint_cmd = spawned[1][1]
    assert "omarchy_cluster.joint_engine" in joint_cmd
    assert "--prefill-url" in joint_cmd and "http://prefill:8090" in joint_cmd
    assert "--prefill-kv" in joint_cmd and "10.0.0.9:8025" in joint_cmd
    assert spawned[2][1] == "http://127.0.0.1:8031"


@pytest.mark.parametrize("missing", ["prefill_url", "prefill_kv"])
def test_serve_joint_requires_prefill_args(missing, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from omarchy_cluster import cli
    args = SimpleNamespace(model=str(tmp_path / "m.gguf"), prefill_url="http://p:1",
                           prefill_kv="10.0.0.9:8025")
    setattr(args, missing, None)
    with pytest.raises(SystemExit):
        cli._serve_joint(args)
